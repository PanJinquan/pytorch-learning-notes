#ifndef CAFFE2_UTILS_MATH_H_
#define CAFFE2_UTILS_MATH_H_
// This is a simple translation from the old Caffe math interfaces. We aim to
// still keep it simple, so all platforms would be able to support it fairly
// easily.

// We include the cblas header here so that we can obtain the macros from cblas.
extern "C" {
#include "caffe2/utils/cblas.h"
}

#ifdef CAFFE2_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif // CAFFE2_USE_ACCELERATE

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math_utils.h"

namespace caffe2 {

class Tensor;

// An empty class as a placeholder for a math function that has no specific
// engine specified.
class DefaultEngine {};

namespace math {

template <typename T, class Context>
void Exp(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Log(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Cos(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Acos(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Sin(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Asin(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Tan(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Atan(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Sinh(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Cosh(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void SinCos(const int N, const T* x, T* ys, T* yc, Context* context);
template <typename T, class Context>
void Tanh(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Abs(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Sqr(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Sqrt(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Rsqrt(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Cube(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Cbrt(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Neg(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Sign(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Not(const int N, const T* x, T* y, Context* context);
template <typename T, class Context>
void Powx(const int N, const T* a, const T b, T* y, Context* context);
template <typename T, class Context>
void Inv(const int N, const T* x, T* y, Context* context);

#define CAFFE2_DECLARE_COMPARE_OP(Comp)                                      \
  template <typename T, class Context>                                       \
  void Comp(const int N, const T* A, const T* B, bool* C, Context* context); \
                                                                             \
  template <typename T, class Context, bool kBroadcast1st = false>           \
  void Rowwise##Comp(                                                        \
      const int rows,                                                        \
      const int cols,                                                        \
      const T* A,                                                            \
      const T* B,                                                            \
      bool* C,                                                               \
      Context* context);                                                     \
                                                                             \
  template <typename T, class Context, bool kBroadcast1st = false>           \
  void Colwise##Comp(                                                        \
      const int rows,                                                        \
      const int cols,                                                        \
      const T* A,                                                            \
      const T* B,                                                            \
      bool* C,                                                               \
      Context* context);                                                     \
                                                                             \
  template <typename T, class Context>                                       \
  void Comp(                                                                 \
      const int A_ndim,                                                      \
      const int* A_dims,                                                     \
      const int B_ndim,                                                      \
      const int* B_dims,                                                     \
      const T* A,                                                            \
      const T* B,                                                            \
      bool* C,                                                               \
      Context* context);

CAFFE2_DECLARE_COMPARE_OP(EQ)
CAFFE2_DECLARE_COMPARE_OP(NE)
CAFFE2_DECLARE_COMPARE_OP(LT)
CAFFE2_DECLARE_COMPARE_OP(LE)
CAFFE2_DECLARE_COMPARE_OP(GT)
CAFFE2_DECLARE_COMPARE_OP(GE)

#undef CAFFE2_DECLARE_COMPARE_OP

#define CAFFE2_DECLARE_BINARY_OP(Func)                                    \
  template <typename T, class Context>                                    \
  void Func(const int N, const T* A, const T* B, T* C, Context* context); \
                                                                          \
  template <typename T, class Context, bool kBroadcast1st = false>        \
  void Rowwise##Func(                                                     \
      const int rows,                                                     \
      const int cols,                                                     \
      const T* A,                                                         \
      const T* B,                                                         \
      T* C,                                                               \
      Context* context);                                                  \
                                                                          \
  template <typename T, class Context, bool kBroadcast1st = false>        \
  void Colwise##Func(                                                     \
      const int rows,                                                     \
      const int cols,                                                     \
      const T* A,                                                         \
      const T* B,                                                         \
      T* C,                                                               \
      Context* context);                                                  \
                                                                          \
  template <typename T, class Context>                                    \
  void Func(                                                              \
      const int A_ndim,                                                   \
      const int* A_dims,                                                  \
      const int B_ndim,                                                   \
      const int* B_dims,                                                  \
      const T* A,                                                         \
      const T* B,                                                         \
      T* C,                                                               \
      Context* context);

CAFFE2_DECLARE_BINARY_OP(Add)
CAFFE2_DECLARE_BINARY_OP(Sub)
CAFFE2_DECLARE_BINARY_OP(Mul)
CAFFE2_DECLARE_BINARY_OP(Div)

CAFFE2_DECLARE_BINARY_OP(And)
CAFFE2_DECLARE_BINARY_OP(Or)
CAFFE2_DECLARE_BINARY_OP(Xor)

CAFFE2_DECLARE_BINARY_OP(BitwiseAnd)
CAFFE2_DECLARE_BINARY_OP(BitwiseOr)
CAFFE2_DECLARE_BINARY_OP(BitwiseXor)

#undef CAFFE2_DECLARE_BINARY_OP

template <typename T, class Context>
void ReduceMin(
    const int N,
    const T* x,
    T* y,
    Tensor* scratch_ptr,
    Context* context);

template <typename T, class Context>
void ReduceMax(
    const int N,
    const T* x,
    T* y,
    Tensor* scratch_ptr,
    Context* context);

template <typename T, class Context>
void ReduceMin(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* Y,
    Context* context);

template <typename T, class Context>
void ReduceMax(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* Y,
    Context* context);

template <typename T, class Context>
void ReduceSum(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* Y,
    Context* context);

template <typename T, class Context>
void ReduceMean(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* Y,
    Context* context);

template <typename T, class Context>
void ReduceL1(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* Y,
    Context* context);

template <typename T, class Context>
void ReduceL2(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* Y,
    Context* context);

// Broadcasts X with X_dims to Y with Y_dims.
template <typename T, class Context>
void Broadcast(
    const int X_ndim,
    const int* X_dims,
    const int Y_ndim,
    const int* Y_dims,
    const T* X,
    T* Y,
    Context* context);

// Computes mean and variance over axes.
template <typename T, class Context>
void Moments(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* mean,
    T* variance,
    Context* context);

// Adds batch sub-tensors elementwise to output. Stripe is the stripe length
// and N is the number of elements to add (size of Y).
template <typename T, class Context>
void AddStripedBatch(
    const int N,
    const T* first,
    T* y,
    const int stripe,
    const int batch,
    Context* context);

// Compute the row-wise max of a N*D matrix X, and write it to a N
// dimensional vector y.
template <typename T, class Context>
void RowwiseMax(const int N, const int D, const T* x, T* y, Context* context);

// Compute the column-wise max of a N*D matrix X, and write it to a D
// dimensional vector y.
template <typename T, class Context>
void ColwiseMax(const int N, const int D, const T* x, T* y, Context* context);

// Elemwise maximum of vector x and vector y. z[i] = max(x[i], y[i])
template <typename T, class Context>
void ElemwiseMax(const int N, const T* x, const T* y, T* z, Context* context);

// Elemwise maximum of vector x and scalar alpha. y[i] = max(x[i], alpha)
template <typename T, class Context>
void Maximum(
    const int N,
    const float alpha,
    const T* x,
    T* y,
    Context* context);

// Transpose tensor X with dims by axes and write the result to tensor Y.
template <typename T, class Context>
void Transpose(
    const int ndim,
    const int* dims,
    const int* axes,
    const T* X,
    T* Y,
    Context* context);

// Decaf gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename T, class Context, class Engine = DefaultEngine>
void Gemm(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const T* A,
    const T* B,
    const float beta,
    T* C,
    Context* context,
    TensorProto::DataType math_type = TensorProto_DataType_FLOAT);

// We also provide a gemm that has explicit lda, ldb and ldc specified.
// In most cases you probably want to use the function above, though.
template <typename T, class Context, class Engine = DefaultEngine>
void GemmEx(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const T alpha,
    const T* A,
    const int lda,
    const T* B,
    const int ldb,
    const T beta,
    T* C,
    const int ldc,
    Context* context);

// GemmBatched provides a simple abstraction into library routines
template <typename T, class Context, class Engine = DefaultEngine>
void GemmBatched(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const T** A,
    const T** B,
    const float beta,
    T** C,
    Context* context,
    TensorProto::DataType math_type = TensorProto_DataType_FLOAT);

template <typename T, class Context, class Engine = DefaultEngine>
void GemmStridedBatched(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const T* A,
    const int A_stride,
    const T* B,
    const int B_stride,
    const float beta,
    T* C,
    const int C_stride,
    Context* context,
    TensorProto::DataType math_type = TensorProto_DataType_FLOAT);

// Gemv always takes in a M*N matrix A, and depending on whether we set TransA
// to Trans, the output is:
// CblasNoTrans: x is an N dim vector and y is an M dim vector.
// CblasTrans:   x is an M dim vector and y is an N dim vector.
template <typename T, class Context, class Engine = DefaultEngine>
void Gemv(
    const CBLAS_TRANSPOSE trans_A,
    const int M,
    const int N,
    const float alpha,
    const T* A,
    const T* x,
    const float beta,
    T* y,
    Context* context,
    TensorProto::DataType math_type = TensorProto_DataType_FLOAT);

template <typename T, class Context>
void Set(const size_t N, const T alpha, T* X, Context* context);

template <typename T, class Context>
void RandUniform(const size_t n, const T a, const T b, T* r, Context* context);

// Generate n values that sum up to a fixed sum
// and subject to a restriction a <= x <= b for each x generated
template <typename T, class Context>
void RandFixedSum(
    const size_t n,
    const T a,
    const T b,
    const T sum,
    T* r,
    Context* context);

template <typename T, class Context>
void RandUniformUnique(
    const size_t n,
    const T a,
    const T b,
    T* r,
    const size_t m,
    const T* avoid,
    Context* context);

template <typename T, class Context>
void RandGaussian(
    const size_t n,
    const T mean,
    const T std,
    T* r,
    Context* context);

// Dot matrix of vector a and b, and writes the result to a single value y.
template <typename T, class Context>
void Dot(const int N, const T* a, const T* b, T* y, Context* context);

// Sum of vector x, and writes the result to a single value y.
template <typename T, class Context>
void Sum(
    const int N,
    const T* x,
    T* y,
    Context* context,
    Tensor* scratch_ptr = nullptr);

// Sum of squares of vector x, and writes the result to a single value y.
template <typename T, class Context>
void SumSqr(
    const int N,
    const T* x,
    T* y,
    Context* context,
    Tensor* scratch_ptr = nullptr);

// Select does index selection of the rows a N*D matrix x, and gives the N
// dimensional vector y that contains the selected data.
template <typename T, class Context>
void Select(
    const int N,
    const int D,
    const T* x,
    const int* idx,
    T* y,
    Context* context);

template <typename T, class Context>
void Scale(const int N, const float alpha, const T* x, T* y, Context* context);

// Different from the Scale function above, if alpha is passed in
// as a pointer, we will assume that it lives on the Context device,
// for example on GPU.
template <typename T, class Context>
void Scale(const int N, const float* alpha, const T* x, T* y, Context* context);

template <typename T, class Context>
void Axpy(const int N, const float alpha, const T* x, T* y, Context* context);

// Different from the Axpy function above, if alpha is passed in
// as a pointer, we will assume that it lives on the Context device,
// for example on GPU.
template <typename T, class Context>
void Axpy(const int N, const float* alpha, const T* x, T* y, Context* context);

template <typename T, class Context>
void Axpby(
    const int N,
    const float alpha,
    const T* x,
    const T b,
    T* y,
    Context* context);

template <typename T, class Context, StorageOrder kOrder>
void Im2ColNd(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const T* img_data,
    T* col_data,
    Context* context);

template <typename T, class Context, StorageOrder kOrder>
void Col2ImNd(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const T* col_data,
    T* img_data,
    Context* context);

template <typename T, class Context, StorageOrder kOrder>
void Im2Col(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const T* img_data,
    T* col_data,
    Context* context);

template <typename T, class Context, int order>
void Col2Im(
    const int channels,
    const int height,
    const int width,
    const int patch_h,
    const int patch_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const T* col_data,
    T* img_data,
    Context* context);

// Applies a per-channel bias value to each channel of the input
// image. image_size is H * W
template <typename T, class Context>
void BiasCHW(
    const T* bias,
    const T* bias_multiplier,
    const int bias_channels,
    const int image_size,
    T* image,
    Context* context);

template <class Context>
void CopyMatrix(
    const size_t item_size,
    const int M,
    const int N,
    const void* A,
    const int lda,
    void* B,
    const int ldb,
    Context* context,
    TypeMeta::TypedCopy copy = nullptr);

template <typename T, class Context>
void CopyMatrix(
    const int M,
    const int N,
    const T* A,
    const int lda,
    T* B,
    const int ldb,
    Context* context);

template <typename T, class Context>
void CopyVector(const int N, const T* A, T* B, Context* context);

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always
// positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than
// 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

// Calculates ceil(a / b). User must be careful to ensure that there
// is no overflow or underflow in the calculation.
template <typename T>
constexpr T divUp(T a, T b) {
  return (a + b - (T)1) / b;
}

// Rounds a up to the next highest multiple of b. User must be careful
// to ensure that there is no overflow or underflow in the calculation
// of divUp.
template <typename T>
constexpr T roundUp(T a, T b) {
  return divUp<T>(a, b) * b;
}

// Returns log2(n) for a positive integer type
template <typename T>
constexpr int integerLog2(T n, int p = 0) {
  return (n <= 1) ? p : integerLog2(n / 2, p + 1);
}

// Returns the next highest power-of-2 for an integer type
template <typename T>
constexpr T integerNextHighestPowerOf2(T v) {
  return (integerIsPowerOf2(v) ? (T)2 * v : ((T)1 << (integerLog2(v) + 1)));
}

} // namespace math
} // namespace caffe2

#include "caffe2/utils/math-detail.h"
#endif // CAFFE2_UTILS_MATH_H_
