#pragma once

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#ifdef __GXX_RTTI
#include <typeinfo>
#endif

#include <exception>

#include "ATen/core/Half.h"
#include "caffe2/core/common.h"
#include "ATen/core/IdWrapper.h"

namespace caffe2 {
class CaffeTypeId;
}

std::ostream& operator<<(std::ostream& stream, caffe2::CaffeTypeId typeId);

namespace caffe2 {

class TypeMeta;

/**
 * A type id is a unique id for a given C++ type.
 * You need to register your types using CAFFE_KNOWN_TYPE(MyType) to be able to use CaffeTypeId with custom types.
 * This is for example used to store the dtype of tensors.
 */
class CaffeTypeId final : public at::IdWrapper<CaffeTypeId, uint16_t> {
public:
  static CaffeTypeId createTypeId();

  friend std::ostream& ::operator<<(std::ostream& stream, CaffeTypeId typeId);
  friend bool operator<(CaffeTypeId lhs, CaffeTypeId rhs);

  // This is 8, because 0 is uint8_t (due to ScalarType BC constraint)
  static constexpr CaffeTypeId uninitialized() {
    return CaffeTypeId(8);
  }

private:
    constexpr explicit CaffeTypeId(uint16_t id): IdWrapper(id) {}
    friend class TypeMeta;
};

// Allow usage in std::map / std::set
// TODO Disallow this and rather use std::unordered_map/set everywhere
inline bool operator<(CaffeTypeId lhs, CaffeTypeId rhs) {
  return lhs.underlyingId() < rhs.underlyingId();
}

}

AT_DEFINE_HASH_FOR_IDWRAPPER(caffe2::CaffeTypeId)

inline std::ostream& operator<<(std::ostream& stream, caffe2::CaffeTypeId typeId) {
  return stream << typeId.underlyingId();
}

namespace caffe2 {

std::unordered_map<CaffeTypeId, std::string>& gTypeNames();
std::unordered_set<std::string>& gRegisteredTypeNames();

// A utility function to demangle a function name.
std::string Demangle(const char* name);

/**
 * Returns the printable name of the type.
 *
 * Works for all types, not only the ones registered with CAFFE_KNOWN_TYPE
 */
template <typename T>
static const char* DemangleType() {
#ifdef __GXX_RTTI
  static const std::string name = Demangle(typeid(T).name());
  return name.c_str();
#else // __GXX_RTTI
  return "(RTTI disabled, cannot show name)";
#endif // __GXX_RTTI
}

// A utility function to return an exception std::string by prepending its exception
// type before its what() content.
std::string GetExceptionString(const std::exception& e);

std::mutex& gTypeRegistrationMutex();

template <typename T>
struct TypeNameRegisterer {
  TypeNameRegisterer(CaffeTypeId id, const std::string& literal_name) {
    std::lock_guard<std::mutex> guard(gTypeRegistrationMutex());
#ifdef __GXX_RTTI
    (void)literal_name;

    std::string name = Demangle(typeid(T).name());
    // If we are in RTTI mode, we will also use this opportunity to do sanity
    // check if there are duplicated ids registered for the same type. This
    // usually happens when one does not do RTLD_GLOBAL, which is often the
    // case in Python. The way we do the check is to make sure that there are
    // no duplicated names registered - this could be done by checking the
    // uniqueness of names.
    if (gRegisteredTypeNames().count(name)) {
      std::cerr << "Type name " << name
                << " registered twice. This should "
                   "not happen. Do you have duplicated CAFFE_KNOWN_TYPE?"
                << std::endl;
      throw std::runtime_error("TypeNameRegisterer error with type " + name);
    }
    gRegisteredTypeNames().insert(name);
    gTypeNames()[id] = name;
#else // __GXX_RTTI
    if (literal_name.empty()) {
      gTypeNames()[id] = "(RTTI disabled, cannot show name)";
    } else {
      gTypeNames()[id] = literal_name;
    }
#endif // __GXX_RTTI
  }
};

/**
 * TypeMeta is a thin class that allows us to store the type of a container such
 * as a blob, or the data type of a tensor, with a unique run-time id. It also
 * stores some additional data such as the item size and the name of the type
 * for run-time inspection.
 */
class TypeMeta {
 public:
  using PlacementNew = void (void*, size_t);
  using TypedCopy = void (const void*, void*, size_t);
  using TypedDestructor = void (void*, size_t);
  /** Create a dummy TypeMeta object. To create a TypeMeta object for a specific
   * type, use TypeMeta::Make<T>().
   */
  TypeMeta() noexcept
      : id_(CaffeTypeId::uninitialized()), itemsize_(0), ctor_(nullptr), copy_(nullptr), dtor_(nullptr) {}

  /**
   * Copy constructor.
   */
  TypeMeta(const TypeMeta& src) noexcept = default;

  /**
   * Assignment operator.
   */
  TypeMeta& operator=(const TypeMeta& src) noexcept = default;

  TypeMeta(TypeMeta &&rhs) noexcept = default;

 private:
  // TypeMeta can only be created by Make, making sure that we do not
  // create incorrectly mixed up TypeMeta objects.
  TypeMeta(
      CaffeTypeId i,
      size_t s,
      PlacementNew* ctor,
      TypedCopy* copy,
      TypedDestructor* dtor) noexcept
      : id_(i), itemsize_(s), ctor_(ctor), copy_(copy), dtor_(dtor) {}

  // Mechanism for throwing errors which can't be prevented at compile time
  // due to type erasure. E.g. somebody calling TypeMeta::copy() for
  // non-copiable type. Right now just throws exception but is implemented
  // in .cpp to manage dependencies
  static void _ThrowRuntimeTypeLogicError(const std::string& msg);

 public:
  /**
   * Returns the type id.
   */
  const CaffeTypeId& id() const noexcept {
    return id_;
  }
  /**
   * Returns the size of the item.
   */
  const size_t& itemsize() const noexcept {
    return itemsize_;
  }
  /**
   * Returns the placement new function pointer for individual items.
   */
  PlacementNew* ctor() const noexcept {
    return ctor_;
  }
  /**
   * Returns the typed copy function pointer for individual iterms.
   */
  TypedCopy* copy() const noexcept {
    return copy_;
  }
  /**
   * Returns the destructor function pointer for individual items.
   */
  TypedDestructor* dtor() const noexcept {
    return dtor_;
  }
  /**
   * Returns a printable name for the type.
   */
  const char* name() const noexcept {
    auto it = gTypeNames().find(id_);
    assert(it != gTypeNames().end());
    return it->second.c_str();
  }

  friend bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept;

  template <typename T>
  bool Match() const {
    return (id_ == Id<T>());
  }

  // Below are static functions that can be called by passing a specific type.

  /**
   * Returns the unique id for the given type T. The id is unique for the type T
   * in the sense that for any two different types, their id are different; for
   * the same type T, the id remains the same over different calls of the
   * function. However, this is not guaranteed over different runs, as the id
   * is generated during run-time. Do NOT serialize the id for storage.
   */
  template <typename T>
  CAFFE2_API static CaffeTypeId Id();

  /**
   * Returns the item size of the type. This is equivalent to sizeof(T).
   */
  template <typename T>
  static size_t ItemSize() {
    return sizeof(T);
  }

  /**
   * Returns the registered printable name of the type.
   *
   * Works for only the ones registered with CAFFE_KNOWN_TYPE
   */
  template <typename T>
  static const char* TypeName() {
    auto it = gTypeNames().find(Id<T>());
    assert(it != gTypeNames().end());
    return it->second.c_str();
  }

  /**
   * Placement new function for the type.
   */
  template <typename T>
  static void _Ctor(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
      new (typed_ptr + i) T;
    }
  }

  template <typename T>
  static void _CtorNotDefault(void* /*ptr*/, size_t /*n*/) {
    _ThrowRuntimeTypeLogicError(
        "Type " + std::string(DemangleType<T>()) +
        " is not default-constructible.");
  }

  template <
      typename T,
      typename std::enable_if<std::is_default_constructible<T>::value>::type* =
          nullptr>
  static inline PlacementNew* _PickCtor() {
    return _Ctor<T>;
  }

  template <
      typename T,
      typename std::enable_if<!std::is_default_constructible<T>::value>::type* =
          nullptr>
  static inline PlacementNew* _PickCtor() {
    return _CtorNotDefault<T>;
  }

  /**
   * Typed copy function for classes.
   */
  template <typename T>
  static void _Copy(const void* src, void* dst, size_t n) {
    const T* typed_src = static_cast<const T*>(src);
    T* typed_dst = static_cast<T*>(dst);
    for (size_t i = 0; i < n; ++i) {
      typed_dst[i] = typed_src[i];
    }
  }

  /**
   * A placeholder function for types that do not allow assignment.
   */
  template <typename T>
  static void
  _CopyNotAllowed(const void* /*src*/, void* /*dst*/, size_t /*n*/) {
    _ThrowRuntimeTypeLogicError(
        "Type " + std::string(DemangleType<T>()) +
        " does not allow assignment.");
  }

  template <
      typename T,
      typename std::enable_if<std::is_copy_assignable<T>::value>::type* =
          nullptr>
  static inline TypedCopy* _PickCopy() {
    return _Copy<T>;
  }

  template <
      typename T,
      typename std::enable_if<!std::is_copy_assignable<T>::value>::type* =
          nullptr>
  static inline TypedCopy* _PickCopy() {
    return _CopyNotAllowed<T>;
  }

  /**
   * Destructor for non-fundamental types.
   */
  template <typename T>
  static void _Dtor(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
      typed_ptr[i].~T();
    }
  }

  /**
   * Returns a TypeMeta object that corresponds to the typename T.
   */
  template <typename T>
  static typename std::enable_if<
      std::is_fundamental<T>::value || std::is_pointer<T>::value,
      TypeMeta>::type
  Make() {
    return TypeMeta(Id<T>(), ItemSize<T>(), nullptr, nullptr, nullptr);
  }

  template <typename T>
  static typename std::enable_if<
      !(std::is_fundamental<T>::value || std::is_pointer<T>::value),
      TypeMeta>::type
  Make() {
    return TypeMeta(
        Id<T>(), ItemSize<T>(), _PickCtor<T>(), _PickCopy<T>(), _Dtor<T>);
  }

 private:
  CaffeTypeId id_;
  size_t itemsize_;
  PlacementNew* ctor_;
  TypedCopy* copy_;
  TypedDestructor* dtor_;
};

inline bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
  return (lhs.id_ == rhs.id_);
}
inline bool operator!=(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
  return !operator==(lhs, rhs);
}

/**
 * Register unique id for a type so it can be used in TypeMeta context, e.g. be
 * used as a type for Blob or for Tensor elements.
 *
 * CAFFE_KNOWN_TYPE does explicit instantiation of TypeMeta::Id<T> template
 * function and thus needs to be put in a single translation unit (.cpp file)
 * for a given type T. Other translation units that use type T as a type of the
 * caffe2::Blob or element type of caffe2::Tensor need to depend on the
 * translation unit that contains CAFFE_KNOWN_TYPE declaration via regular
 * linkage dependencies.
 *
 * NOTE: the macro needs to be invoked in ::caffe2 namespace
 */
// Implementation note: in MSVC, we will need to prepend the CAFFE2_EXPORT
// keyword in order to get things compiled properly. in Linux, gcc seems to
// create attribute ignored error for explicit template instantiations, see
//   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0537r0.html
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51930
// and as a result, we define these two macros slightly differently.

#ifdef _MSC_VER
#define CAFFE_KNOWN_TYPE(T)                                                        \
  template <>                                                                      \
  CAFFE2_EXPORT CaffeTypeId TypeMeta::Id<T>() {                                    \
    static const CaffeTypeId type_id = CaffeTypeId::createTypeId();                \
    static TypeNameRegisterer<T> registerer(type_id, #T);                          \
    return type_id;                                                                \
  }
#else // _MSC_VER
#define CAFFE_KNOWN_TYPE(T)                                                        \
  template <>                                                                      \
  CaffeTypeId TypeMeta::Id<T>() {                                                  \
    static const CaffeTypeId type_id = CaffeTypeId::createTypeId();                \
    static TypeNameRegisterer<T> registerer(type_id, #T);                          \
    return type_id;                                                                \
  }
#endif

/**
 * CAFFE_DECLARE_KNOWN_TYPE and CAFFE_DEFINE_KNOWN_TYPE are used
 * to preallocate ids for types that are queried very often so that they
 * can be resolved at compile time. Please use CAFFE_KNOWN_TYPE() instead
 * for your own types to allocate dynamic ids for them.
 */
#ifdef _MSC_VER
#define CAFFE_DECLARE_KNOWN_TYPE(PreallocatedId, T)              \
  template <>                                                    \
  inline CAFFE2_EXPORT CaffeTypeId TypeMeta::Id<T>() { \
    return CaffeTypeId(PreallocatedId);                          \
  }
#else // _MSC_VER
#define CAFFE_DECLARE_KNOWN_TYPE(PreallocatedId, T) \
  template <>                                       \
  inline CaffeTypeId TypeMeta::Id<T>() {  \
    return CaffeTypeId(PreallocatedId);             \
  }
#endif

#define CONCAT_IMPL(x, y) x##y
#define MACRO_CONCAT(x, y) CONCAT_IMPL(x, y)

#define CAFFE_DEFINE_KNOWN_TYPE(T)                             \
  namespace {                                                  \
  TypeNameRegisterer<T> MACRO_CONCAT(registerer, __COUNTER__)( \
      TypeMeta::Id<T>(),                                       \
      #T);                                                     \
  }

class Tensor;

// Note: we have preallocated the numbers 0-8 so they line up exactly
// with at::ScalarType's numbering.  All other numbers do not matter.
//
// Notably, the "uninitialized" type id is 8, not 0, for hysterical raisins.

struct _CaffeHighestPreallocatedTypeId final {};

CAFFE_DECLARE_KNOWN_TYPE(0, uint8_t);
CAFFE_DECLARE_KNOWN_TYPE(1, int8_t);
CAFFE_DECLARE_KNOWN_TYPE(2, int16_t);
CAFFE_DECLARE_KNOWN_TYPE(3, int);
CAFFE_DECLARE_KNOWN_TYPE(4, int64_t);
CAFFE_DECLARE_KNOWN_TYPE(5, at::Half);
CAFFE_DECLARE_KNOWN_TYPE(6, float);
CAFFE_DECLARE_KNOWN_TYPE(7, double);
// 8 = undefined type id

CAFFE_DECLARE_KNOWN_TYPE(9, Tensor);
CAFFE_DECLARE_KNOWN_TYPE(10, std::string);
CAFFE_DECLARE_KNOWN_TYPE(11, bool);
CAFFE_DECLARE_KNOWN_TYPE(12, uint16_t);
CAFFE_DECLARE_KNOWN_TYPE(13, char);
CAFFE_DECLARE_KNOWN_TYPE(14, std::unique_ptr<std::mutex>);
CAFFE_DECLARE_KNOWN_TYPE(15, std::unique_ptr<std::atomic<bool>>);
CAFFE_DECLARE_KNOWN_TYPE(16, std::vector<int32_t>);
CAFFE_DECLARE_KNOWN_TYPE(17, std::vector<int64_t>);
CAFFE_DECLARE_KNOWN_TYPE(18, std::vector<unsigned long>);
CAFFE_DECLARE_KNOWN_TYPE(19, bool*);
CAFFE_DECLARE_KNOWN_TYPE(20, char*);
CAFFE_DECLARE_KNOWN_TYPE(21, int*);

#ifdef CAFFE2_UNIQUE_LONG_TYPEMETA
CAFFE_DECLARE_KNOWN_TYPE(22, long);
CAFFE_DECLARE_KNOWN_TYPE(23, std::vector<long>);
#endif // CAFFE2_UNIQUE_LONG_TYPEMETA

CAFFE_DECLARE_KNOWN_TYPE(24, _CaffeHighestPreallocatedTypeId);
}
