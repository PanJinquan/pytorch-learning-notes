#ifndef CAFFE2_CORE_BLOB_H_
#define CAFFE2_CORE_BLOB_H_

#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include "caffe2/core/blob_serializer_base.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/typeid.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

/**
 * @brief Blob is a general container that hosts a typed pointer.
 *
 * A Blob hosts a pointer as well as its type, and takes charge of deleting it
 * properly when the blob is deallocated or re-allocated with a new type. A blob
 * could contain anything, although the most common case is to contain a Tensor.
 */
class Blob {
 public:
  typedef void (*DestroyCall)(void*);

  /**
   * Initializes an empty Blob.
   */
  Blob() : meta_(), pointer_(nullptr) {}
  ~Blob() { Reset(); }

  Blob(Blob&& other) noexcept
      : meta_(std::move(other.meta_)),
        pointer_(std::move(other.pointer_)),
        destroy_(std::move(other.destroy_)) {
    other.meta_ = {};
    other.pointer_ = nullptr;
    other.destroy_ = nullptr;
  }

  Blob& operator=(Blob&& other) noexcept {
    if (pointer_ && destroy_) {
      destroy_(pointer_);
    }
    meta_ = std::move(other.meta_);
    pointer_ = std::move(other.pointer_);
    destroy_ = std::move(other.destroy_);
    other.meta_ = {};
    other.pointer_ = nullptr;
    other.destroy_ = nullptr;
    return *this;
  }

  /**
   * Checks if the content stored in the blob is of type T.
   */
  template <class T>
  bool IsType() const { return meta_.Match<T>(); }

  // TODO(jerryzh): Remove template
  template <class T>
  bool IsType(DeviceType device_type) const {
    static_assert(
        std::is_same<T, Tensor>::value,
        "IsType(DeviceType) only available on "
        "Tensor types.");
    auto* tensor = static_cast<Tensor*>(pointer_);
    if (tensor && tensor->GetDeviceType() == device_type) {
      return true;
    }
    return false;
  }

  /**
   * Returns the meta info of the blob.
   */
  inline const TypeMeta& meta() const { return meta_; }

  /**
   * Returns a printable typename of the blob.
   */
  inline const char* TypeName() const { return meta_.name(); }

  /**
   * @brief Gets the const reference of the stored object. The code checks if
   * the stored object is of the desired type.
   */
  // TODO(jerryzh): add a Get(DeviceType) function?
  template <class T>
  const T& Get() const {
    CAFFE_ENFORCE(
        IsType<T>(),
        "wrong type for the Blob instance. Blob contains ",
        meta_.name(),
        " while caller expects ",
        TypeMeta::TypeName<T>());
    return *static_cast<const T*>(pointer_);
  }

  const void* GetRaw() const {
    return pointer_;
  }
  void* GetRaw() {
    return pointer_;
  }

  /**
   * @brief Gets a mutable pointer to the stored object.
   *
   * If the current object is not of the right type, a new object is created
   * and the old object is freed. Note that type T should have a default
   * constructor. Otherwise, create the object yourself first, and use
   * Reset().
   */
  template <class T>
  T* GetMutable() {
    static_assert(
        std::is_default_constructible<T>::value,
        "GetMutable can't be called with non-default-constructible types. "
        "Try using specialized methods");
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<T>();
      return Reset<T>(new T());
    }
  }

  template <class T>
  T* GetMutableOrNull() {
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      return nullptr;
    }
  }

  inline Tensor* GetMutableTensor(DeviceType device_type) {
    if (IsType<Tensor>() &&
        static_cast<Tensor*>(pointer_)->GetDeviceType() == device_type) {
      return static_cast<Tensor*>(pointer_);
    } else {
      VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<Tensor>()
              << " DeviceType:" << device_type;
      return Reset<Tensor>(new Tensor(device_type));
    }
  }

  /**
   * Sets the underlying object to the allocated one. The Blob then takes over
   * the ownership of the passed in pointer. If there is already an object in
   * the Blob, the old object is freed.
   *
   * This is used when the underlying class T does not have a default ctor, or
   * complex initializations needs to be done outside the blob.
   */
  template <class T>
  T* Reset(T* allocated) {
    if (pointer_ && destroy_) {
      destroy_(pointer_);
    }
    meta_ = TypeMeta::Make<T>();
    pointer_ = static_cast<void*>(allocated);
    destroy_ = &Destroy<T>;
    return allocated;
  }

  inline void*
  Reset(void* allocated, const TypeMeta& meta, const DestroyCall& destroy) {
    if (pointer_ && destroy_) {
      destroy_(pointer_);
    }
    meta_ = meta;
    pointer_ = static_cast<void*>(allocated);
    destroy_ = destroy;
    return allocated;
  }

  /**
   * Releases the ownership, if any, this Blob has on the underlying pointer.
   * The user is then responsible for freeing the data if needed
   */
  inline DestroyCall Release() {
    DestroyCall d = destroy_;
    destroy_ = nullptr;
    return d;
  }

  /**
   * Sets the underlying object to the allocated one, but does not take over
   * the ownership of the passed in pointer. If there is already an object in
   * the Blob, the old object is freed.
   *
   * Unlike Reset, this does not take over the ownership of the pointer and the
   * caller is responsible for making sure that the lifetime of the allocated
   * blob outlasts the lifetime of any access to this blob, until another Reset
   * call is made or the blob is destructed.
   */
  template <class T>
  typename std::remove_const<T>::type* ShareExternal(
      typename std::remove_const<T>::type* allocated) {
    return static_cast<T*>(ShareExternal(
        static_cast<void*>(allocated),
        TypeMeta::Make<typename std::remove_const<T>::type>()));
  }

  void* ShareExternal(void* allocated, const TypeMeta& meta) {
    if (pointer_ && destroy_) {
      destroy_(pointer_);
    }
    meta_ = meta;
    pointer_ = static_cast<void*>(allocated);
    destroy_ = nullptr;
    return allocated;
  }

  /**
   * Resets the Blob to an empty one.
   */
  inline void Reset() {
    if (pointer_ && destroy_) {
      destroy_(pointer_);
    }
    pointer_ = nullptr;
    meta_ = TypeMeta();
    destroy_ = nullptr;
  }

  /**
   * Serializes the current blob, if possible. Note that this serialization uses
   * the registration mechanism and one has to implement specific serialization
   * approaches for specific classes. Acceptor should take care of writing data
   * to the actual storage.
   */
  void Serialize(
      const string& name,
      BlobSerializerBase::SerializationAcceptor acceptor,
      int chunk_size = kDefaultChunkSize) const;

  /**
   * @brief Convenience function to serialize a blob to a string.
   *
   * This is a conveinence function to serialize small Blobs that produce
   * manageable serialized strings. To serialize big blobs such as
   * large sparse tensors, use the fully-functional interface in
   * blob_serializer_base.h.
   *
   * NOTE: this function doesn't do chunking and might break with big tensors.
   */
  string Serialize(const string& name) const;

  /**
   * @brief Swaps the underlying storage of two blobs.
   */
  void swap(Blob& rhs) {
    using std::swap;
    swap(meta_, rhs.meta_);
    swap(pointer_, rhs.pointer_);
    swap(destroy_, rhs.destroy_);
  }

  /**
   * Deserializes from a string containing either BlobProto or TensorProto. If
   * the deserialization fails, the content in the blob should no longer be
   * trusted.
   */
  void Deserialize(const string& content);
  void Deserialize(const BlobProto& proto);

 private:
  /**
   * @brief A destroy call that is used to properly deconstruct objects.
   */
  template <class T>
  static void Destroy(void* pointer) {
    delete static_cast<T*>(pointer);
  }
  TypeMeta meta_;
  void* pointer_ = nullptr;
  DestroyCall destroy_ = nullptr;

  DISABLE_COPY_AND_ASSIGN(Blob);
};

inline void swap(Blob& lhs, Blob& rhs) {
  lhs.swap(rhs);
}

}  // namespace caffe2
#endif  // CAFFE2_CORE_BLOB_H_
