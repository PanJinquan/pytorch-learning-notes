#ifndef CAFFE2_CORE_CONTEXT_H_
#define CAFFE2_CORE_CONTEXT_H_

#include <cstdlib>
#include <ctime>
#include <random>
#include <unordered_map>

#include "caffe2/core/allocator.h"
#include "caffe2/core/context_base.h"
#include "caffe2/core/event.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/typeid.h"
#include "caffe2/proto/caffe2.pb.h"

#include "ATen/core/ATenCoreTest.h"
#include "ATen/core/ArrayRef.h"

CAFFE2_DECLARE_bool(caffe2_report_cpu_memory_usage);

namespace caffe2 {

BaseStaticContext* GetCPUStaticContext();

/**
 * A function to generate a random number seed that is unique in a best-effort
 * basis, using an ever-incrementing seed and the current time.
 */
uint32_t RandomNumberSeed();

/**
 * The CPU Context, representing the bare minimum of what a Context class in
 * Caffe2 should implement.
 *
 * // TODO modify docs
 * See operator.h, especially Operator<Context>, for how Context are used in
 * actual operator implementations that are associated with specific devices.
 * In general, the Context class is passed in as a template argument, and
 * the operator can use the functions defined in the context to execute whatever
 * computation it has.
 *
 */
class CPUContext final : public BaseContext {
 public:
  typedef std::mt19937 rand_gen_type;
  CPUContext() : random_seed_(RandomNumberSeed()) {}
  explicit CPUContext(const DeviceOption& option)
      : random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : RandomNumberSeed()) {
    CAFFE_ENFORCE_EQ(option.device_type(), CPU);
  }

  ~CPUContext() noexcept override {}

  BaseStaticContext* GetStaticContext() const override {
    return GetCPUStaticContext();
  }

  static BaseStaticContext* StaticContext() {
    return GetCPUStaticContext();
  }

  inline void SwitchToDevice(int /*stream_id*/) override {}

  using BaseContext::SwitchToDevice;

  inline void WaitEvent(const Event& ev) override {
    ev.Wait(CPU, this);
  }

  inline void Record(Event* ev, const char* err_msg = nullptr) const override {
    CAFFE_ENFORCE(ev, "Event must not be null.");
    ev->Record(CPU, this, err_msg);
  }

  inline void FinishDeviceComputation() override {}

  inline rand_gen_type& RandGenerator() {
    if (!random_generator_.get()) {
      random_generator_.reset(new rand_gen_type(random_seed_));
    }
    return *random_generator_.get();
  }

  inline static std::pair<void*, MemoryDeleter> New(size_t nbytes) {
    return StaticContext()->New(nbytes);
  }

  void CopyBytesSameDevice(size_t nbytes, const void* src, void* dst) override {
    if (nbytes == 0) {
      return;
    }
    CAFFE_ENFORCE(src);
    CAFFE_ENFORCE(dst);
    memcpy(dst, src, nbytes);
  }

  void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) override {
    CopyBytesSameDevice(nbytes, src, dst);
  }

  void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) override {
    CopyBytesSameDevice(nbytes, src, dst);
  }

  bool SupportsNonFundamentalTypes() const override {
    // CPU non fumdamental type copy OK
    return true;
  }

  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void* src, void* dst);

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(size_t n, const T* src, T* dst) {
    if (std::is_fundamental<T>::value) {
      CopyBytes<SrcContext, DstContext>(
          n * sizeof(T),
          static_cast<const void*>(src),
          static_cast<void*>(dst));
    } else {
      for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
      }
    }
  }

  template <class SrcContext, class DstContext>
  inline void
  CopyItems(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      meta.copy()(src, dst, n);
    } else {
      CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
    }
  }

  // By default CPU operators don't have async device parts
  static bool HasAsyncPartDefault() {
    return false;
  }

  static bool SupportsAsyncScheduling() {
    return false;
  }

  // CPU streams are not implemented and are silently ignored by CPU ops,
  // return true to signal executor to schedule a CPU op
  static bool IsStreamFree(
      const DeviceOption& /* option */,
      int /* stream_id */) {
    return true;
  }

  DeviceType GetDevicetype() const override {
    return CPU;
  }

  static constexpr DeviceType GetDeviceType() {
    return CPU;
  }

 protected:
  // TODO(jiayq): instead of hard-coding a generator, make it more flexible.
  int random_seed_{1701};
  std::unique_ptr<rand_gen_type> random_generator_;
};

template <>
inline void CPUContext::CopyBytes<CPUContext, CPUContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  if (nbytes == 0) {
    return;
  }
  CAFFE_ENFORCE(src);
  CAFFE_ENFORCE(dst);
  memcpy(dst, src, nbytes);
}

// TODO(jerryzh): merge CPUStaticContext with Allocator
class CPUStaticContext : public BaseStaticContext {
 public:
  std::pair<void*, MemoryDeleter> New(size_t nbytes) const override {
    auto data_and_deleter = GetCPUAllocator()->New(nbytes);
    if (FLAGS_caffe2_report_cpu_memory_usage) {
      reporter_.New(data_and_deleter.first, nbytes);
      data_and_deleter.second = ReportAndDelete;
    }
    return data_and_deleter;
  }

  std::unique_ptr<BaseContext> CreateContext() override {
    return caffe2::make_unique<CPUContext>();
  }

  std::unique_ptr<BaseContext> CreateContext(
      const DeviceOption& option) override {
    return caffe2::make_unique<CPUContext>(option);
  }

  DeviceType GetDeviceType() override {
    return CPU;
  }

 protected:
  CAFFE2_API static MemoryAllocationReporter reporter_;

 private:
  static void ReportAndDelete(void* ptr) {
    reporter_.Delete(ptr);
    GetCPUAllocator()->GetDeleter()(ptr);
  }
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_CONTEXT_H_
