/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkOpenCLKernel_h
#define itkOpenCLKernel_h

#include "itkOpenCL.h"
#include "itkOpenCLGlobal.h"
#include "itkOpenCLEvent.h"
#include "itkOpenCLSize.h"
#include "itkOpenCLMemoryObject.h"
#include "itkOpenCLSampler.h"
#include "itkOpenCLVector.h"

#include "itkIndex.h"
#include "itkPoint.h"
#include "itkVector.h"
#include "itkCovariantVector.h"
#include "itkMatrix.h"

namespace itk
{
/**
 * \class OpenCLKernel
 * \brief The OpenCLKernel class represents an executable entry point
 * function in an OpenCL program.
 *
 * \section1 Executing kernels
 * OpenCLKernel corresponds to an instance of an OpenCL kernel, decorated
 * with a specific GetGlobalWorkSize() and GetLocalWorkSize(). It is possible
 * to use the same OpenCL kernel with different work sizes by altering
 * the size for each execution request:
 *
 * \code
 * OpenCLKernel kernel = program.CreateKernel("foo");
 * kernel.SetGlobalWorkSize(100, 100);
 * kernel.SetArg(0, a1);
 * kernel.SetArg(1, b1);
 * kernel.LaunchKernel();
 * kernel.SetGlobalWorkSize(200, 200);
 * kernel.SetArg(0, a2);
 * kernel.SetArg(1, b2);
 * kernel.LaunchKernel();
 * \endcode
 *
 * Alternatively, operator()() can be used to avoid the SetArg() calls:
 *
 * \code
 * OpenCLKernel kernel = program.CreateKernel("foo");
 * kernel.SetGlobalWorkSize(100, 100);
 * kernel(a1, b1);
 * kernel.SetGlobalWorkSize(200, 200);
 * kernel(a2, b2);
 * \endcode
 *
 * Up to 10 arguments can be provided to operator()(). Use explicit
 * SetArg() and LaunchKernel() calls with kernels that have more than
 * 10 arguments.
 * The following types are handled specially via setArg() and operator()():
 * \c cl_int(n), \c cl_uint(n), \c cl_long(n), \c cl_ulong(n), \c float(n),
 * Index, Point, Vector, CovariantVector, Matrix,
 * OpenCLBuffer, OpenCLImage, OpenCLVector, and OpenCLSampler.
 * Other argument types must be set explicitly by calling the
 * SetArg() override that takes a buffer and size.
 * \section1 Asynchronous execution
 * Note that both run() and operator()() return immediately;
 * they will not block until execution is finished. Both functions
 * return a OpenCLEvent object that can be used to wait for the
 * request to finish:
 *
 * \code
 * kernel.SetGlobalWorkSize(100, 100);
 * OpenCLEvent event = kernel(a1, b1);
 * event.WaitForFinished();
 * \endcode
 *
 * Usually it isn't necessary for an explicit OpenCLEvent wait
 * because the next OpenCL request will implicitly block until
 * the kernel finishes execution:
 *
 * \code
 * OpenCLBuffer buffer = ...;
 * kernel.SetGlobalWorkSize(100, 100);
 * kernel(buffer);
 * buffer.Read(...);
 * \endcode
 *
 * With the default in-order command execution policy, OpenCL will ensure
 * that the OpenCLBuffer::Read() request will not begin execution until the
 * kernel execution finishes.
 *
 * \ingroup OpenCL
 * \sa OpenCLProgram
 */

// Defines for macro injections to reduce spoil code
#define OpenCLKernelSetArgMacroH(type) cl_int SetArg(const cl_uint index, const type value);

#define OpenCLKernelSetArgMacroCXX(type)                                                                               \
  cl_int OpenCLKernel::SetArg(const cl_uint index, const type value)                                                   \
  {                                                                                                                    \
    return clSetKernelArg(this->m_KernelId, index, sizeof(value), (const void *)&value);                               \
  }

#define OpenCLKernelSetArgsMacroH(type0, type1, type2, type3, type4)                                                   \
  OpenCLKernelSetArgMacroH(type0) OpenCLKernelSetArgMacroH(type1) OpenCLKernelSetArgMacroH(type2)                      \
    OpenCLKernelSetArgMacroH(type3) OpenCLKernelSetArgMacroH(type4)

#define OpenCLKernelSetArgsMacroCXX(type0, type1, type2, type3, type4)                                                 \
  OpenCLKernelSetArgMacroCXX(type0) OpenCLKernelSetArgMacroCXX(type1) OpenCLKernelSetArgMacroCXX(type2)                \
    OpenCLKernelSetArgMacroCXX(type3) OpenCLKernelSetArgMacroCXX(type4)

// Forward declaration
class OpenCLContext;
class OpenCLProgram;
class OpenCLVectorBase;
class OpenCLDevice;

class OpenCLKernelPimpl; // OpenCLKernel private implementation idiom.

class ITKOpenCL_EXPORT OpenCLKernel
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLMemoryObject;

  /** Constructs a null OpenCL kernel object. */
  OpenCLKernel();

  /** Constructs an OpenCL kernel object from the native identifier \a id,
   * and associates it with \a GetContext.  This class will take over
   * ownership of \a id and release it in the destructor. */
  OpenCLKernel(OpenCLContext * context, const cl_kernel id);

  /** Constructs a copy of \a other. */
  OpenCLKernel(const OpenCLKernel & other);

  /** Releases this OpenCL kernel object. If this is the last
   * reference to the kernel, it will be destroyed. */
  ~OpenCLKernel();

  /** Assigns \a other to this object. */
  OpenCLKernel &
  operator=(const OpenCLKernel & other);

  /** Returns true if this OpenCL kernel object is null, false otherwise. */
  bool
  IsNull() const;

  /** Returns the native OpenCL identifier for this kernel. */
  cl_kernel
  GetKernelId() const;

  /** Returns the OpenCL GetContext that this kernel was created within. */
  OpenCLContext *
  GetContext() const;

  /** Returns the OpenCL program that this kernel is associated with. */
  OpenCLProgram
  GetProgram() const;

  /** Returns the name of this OpenCL kernel's entry point function. */
  std::string
  GetName() const;

  /** Returns the number of arguments that are expected by this OpenCL kernel.
   * \sa setArg() */
  std::size_t
  GetNumberOfArguments() const;

  /** Returns the work group size that was declared in the kernel's
   * source code using a \c{reqd_work_group_size} qualifier.
   * Returns (0, 0, 0) if the size is not declared.
   * The default device for GetContext() is used to retrieve the
   * work group size. */
  OpenCLSize
  GetCompileWorkGroupSize() const;

  /** \overload
   * Returns the work group size that was declared in the kernel's
   * source code using a \c{reqd_work_group_size} qualifier.
   * Returns (0, 0, 0) if the size is not declared.
   * The specified \a device is used to retrieve the work group size. */
  OpenCLSize
  GetCompileWorkGroupSize(const OpenCLDevice & device) const;

  /** Sets the global work size for this instance of the kernel to \a size.
   * \sa GetGlobalWorkSize(), SetLocalWorkSize(), SetRoundedGlobalWorkSize() */
  void
  SetGlobalWorkSize(const OpenCLSize & size);

  /** Returns the global work size for this instance of the kernel.
   * The default value is 1.
   * \sa SetGlobalWorkSize(), GetLocalWorkSize() */
  OpenCLSize
  GetGlobalWorkSize() const;

  /** Sets the global work size for this instance of the kernel to \a size,
   * after rounding it up to the next multiple of GetLocalWorkSize().
   * \sa GetGlobalWorkSize(), OpenCLSize::RoundTo() */
  void
  SetRoundedGlobalWorkSize(const OpenCLSize & size);

  /** Sets the local work size for this instance of the kernel to \a size.
   * \sa SetGlobalWorkSize() */
  void
  SetLocalWorkSize(const OpenCLSize & size);

  /** Sets the offset for this instance of the kernel to \a offset.
   * \sa GetLocalWorkSize() */
  void
  SetGlobalWorkOffset(const OpenCLSize & offset);

  /** Returns the global work offset size for this instance of the kernel.
   * The default value is 0.
   * \sa SetGlobalWorkOffset() */
  OpenCLSize
  GetGlobalWorkOffset() const;

  /** Returns the local work size for this instance of the kernel.
   * The default value is 0, which indicates that the local
   * work size is not used.
   * \sa SetLocalWorkSize(), GetGlobalWorkSize() */
  OpenCLSize
  GetLocalWorkSize() const;

  /** Returns the recommended best local work size for 1D image processing
   * on this kernel. Default value is 8 unless the maximum work size
   * is not large enough to accommodate 8 items.
   * \sa GetBestLocalWorkSizeImage2D(), GetBestLocalWorkSizeImage3D */
  OpenCLSize
  GetBestLocalWorkSizeImage1D() const;

  /** Returns the recommended best local work size for 2D image processing
   * on this kernel. Default value is 8x8 unless the maximum work size
   * is not large enough to accommodate 8x8 items.
   * \sa GetBestLocalWorkSizeImage1D, GetBestLocalWorkSizeImage3D() */
  OpenCLSize
  GetBestLocalWorkSizeImage2D() const;

  /** Returns the recommended best local work size for 3D image processing
   * on this kernel. Default value is 8x8x8 unless the maximum work size
   * is not large enough to accommodate 8x8x8 items.
   * \sa GetBestLocalWorkSizeImage1D, GetBestLocalWorkSizeImage2D() */
  OpenCLSize
  GetBestLocalWorkSizeImage3D() const;

  /** Returns the recommended best local work size for 1D/2D/3D image processing
   * based on the \a dimension on this kernel. */
  OpenCLSize
  GetBestLocalWorkSizeImage(const std::size_t dimension) const;

  /** Returns the preferred work group size multiple, which is a
   * performance hint for the local work group size on OpenCL 1.1
   * systems. Returns zero if the system is OpenCL 1.0, or a
   * preferred work group size multiple is not available. */
  size_t
  GetPreferredWorkSizeMultiple() const;

  /** Sets the flag double as float to \a value. Default is true.
   * By enabling this flag, setting kernel with all double values are
   * converted to float.
   * \not Doesn't effect SetArg(index, double(n)) method.
   * \sa GetDoubleAsFloatEnabled() */
  void
  SetDoubleAsFloat(const bool value)
  {
    this->m_DoubleAsFloat = value;
  }
  void
  SetDoubleAsFloatEnable()
  {
    this->m_DoubleAsFloat = true;
  }
  void
  SetDoubleAsFloatDisable()
  {
    this->m_DoubleAsFloat = false;
  }
  bool
  GetDoubleAsFloatEnabled()
  {
    return this->m_DoubleAsFloat;
  }

  /** Sets argument \a index for this kernel to \a value.
   * The argument is assumed to have been declared with the types:
   * char(n), uchar(n), short(n), ushort(n), int(n), uint(n),
   * long(n), ulong(n), float(n), double(n).
   * Due to number of overloads below, we have to use C++11
   * std::enable_if, std::is_scalar, std::is_union, which doesn't
   * supported by all compilers. */
  /*
  template< typename T,
  typename = typename std::enable_if<
    std::is_scalar< T >::value || std::is_union< T >::value >::type >
  cl_int SetArg( const cl_uint index, const T value )
  {
    return clSetKernelArg( this->m_KernelId, index,
      sizeof( value ), (const void *)&value );
  }
  */

  /** Sets argument \a index for this kernel to \a value.
   * Macro definitions of SetArg() methods for types:
   * char(n), uchar(n), short(n), ushort(n), int(n), uint(n),
   * long(n), ulong(n), float(n), double(n).
   * \note Instead of using macro, we could just use templated version
   * of the SetArg() with C++11 here (see commented code above). */
  OpenCLKernelSetArgsMacroH(cl_uchar, cl_uchar2, cl_uchar4, cl_uchar8, cl_uchar16)
    OpenCLKernelSetArgsMacroH(cl_char, cl_char2, cl_char4, cl_char8, cl_char16)
      OpenCLKernelSetArgsMacroH(cl_ushort, cl_ushort2, cl_ushort4, cl_ushort8, cl_ushort16)
        OpenCLKernelSetArgsMacroH(cl_short, cl_short2, cl_short4, cl_short8, cl_short16)
          OpenCLKernelSetArgsMacroH(cl_uint, cl_uint2, cl_uint4, cl_uint8, cl_uint16)
            OpenCLKernelSetArgsMacroH(cl_int, cl_int2, cl_int4, cl_int8, cl_int16)
              OpenCLKernelSetArgsMacroH(cl_ulong, cl_ulong2, cl_ulong4, cl_ulong8, cl_ulong16)
                OpenCLKernelSetArgsMacroH(cl_long, cl_long2, cl_long4, cl_long8, cl_long16)
                  OpenCLKernelSetArgsMacroH(cl_float, cl_float2, cl_float4, cl_float8, cl_float16)
                    OpenCLKernelSetArgsMacroH(cl_double, cl_double2, cl_double4, cl_double8, cl_double16)

    /** Sets argument \a index to the \a size bytes at \a data. */
    cl_int SetArg(const cl_uint index, const void * data, const size_t size);

  /** Sets argument \a index for this kernel to size \a value. */
  using Size1DType = Size<1>;
  cl_int
  SetArg(const cl_uint index, const Size1DType & value);

  using Size2DType = Size<2>;
  cl_int
  SetArg(const cl_uint index, const Size2DType & value);

  using Size3DType = Size<3>;
  cl_int
  SetArg(const cl_uint index, const Size3DType & value);

  using Size4DType = Size<4>;
  cl_int
  SetArg(const cl_uint index, const Size4DType & value);

  /** Sets argument \a index for this kernel to index \a value. */
  using Index1DType = Index<1>;
  cl_int
  SetArg(const cl_uint index, const Index1DType & value);

  using Index2DType = Index<2>;
  cl_int
  SetArg(const cl_uint index, const Index2DType & value);

  using Index3DType = Index<3>;
  cl_int
  SetArg(const cl_uint index, const Index3DType & value);

  using Index4DType = Index<4>;
  cl_int
  SetArg(const cl_uint index, const Index4DType & value);

  /** Sets argument \a index for this kernel to offset \a value. */
  using Offset1DType = Offset<1>;
  cl_int
  SetArg(const cl_uint index, const Offset1DType & value);

  using Offset2DType = Offset<2>;
  cl_int
  SetArg(const cl_uint index, const Offset2DType & value);

  using Offset3DType = Offset<3>;
  cl_int
  SetArg(const cl_uint index, const Offset3DType & value);

  using Offset4DType = Offset<4>;
  cl_int
  SetArg(const cl_uint index, const Offset4DType & value);

  /** Sets argument \a index for this kernel to point \a value. */
  using PointInt1DType = Point<int, 1>;
  cl_int
  SetArg(const cl_uint index, const PointInt1DType & value);

  using PointFloat1DType = Point<float, 1>;
  cl_int
  SetArg(const cl_uint index, const PointFloat1DType & value);

  using PointDouble1DType = Point<double, 1>;
  cl_int
  SetArg(const cl_uint index, const PointDouble1DType & value);

  using PointInt2DType = Point<int, 2>;
  cl_int
  SetArg(const cl_uint index, const PointInt2DType & value);

  using PointFloat2DType = Point<float, 2>;
  cl_int
  SetArg(const cl_uint index, const PointFloat2DType & value);

  using PointDouble2DType = Point<double, 2>;
  cl_int
  SetArg(const cl_uint index, const PointDouble2DType & value);

  using PointInt3DType = Point<int, 3>;
  cl_int
  SetArg(const cl_uint index, const PointInt3DType & value);

  using PointFloat3DType = Point<float, 3>;
  cl_int
  SetArg(const cl_uint index, const PointFloat3DType & value);

  using PointDouble3DType = Point<double, 3>;
  cl_int
  SetArg(const cl_uint index, const PointDouble3DType & value);

  using PointInt4DType = Point<int, 4>;
  cl_int
  SetArg(const cl_uint index, const PointInt4DType & value);

  using PointFloat4DType = Point<float, 4>;
  cl_int
  SetArg(const cl_uint index, const PointFloat4DType & value);

  using PointDouble4DType = Point<double, 4>;
  cl_int
  SetArg(const cl_uint index, const PointDouble4DType & value);

  /** Sets argument \a index for this kernel to vector \a value. */
  /** ITK Vector typedefs. */
  using VectorInt1DType = Vector<int, 1>;
  cl_int
  SetArg(const cl_uint index, const VectorInt1DType & value);

  using VectorFloat1DType = Vector<float, 1>;
  cl_int
  SetArg(const cl_uint index, const VectorFloat1DType & value);

  using VectorDouble1DType = Vector<double, 1>;
  cl_int
  SetArg(const cl_uint index, const VectorDouble1DType & value);

  using VectorInt2DType = Vector<int, 2>;
  cl_int
  SetArg(const cl_uint index, const VectorInt2DType & value);

  using VectorFloat2DType = Vector<float, 2>;
  cl_int
  SetArg(const cl_uint index, const VectorFloat2DType & value);

  using VectorDouble2DType = Vector<double, 2>;
  cl_int
  SetArg(const cl_uint index, const VectorDouble2DType & value);

  using VectorInt3DType = Vector<int, 3>;
  cl_int
  SetArg(const cl_uint index, const VectorInt3DType & value);

  using VectorFloat3DType = Vector<float, 3>;
  cl_int
  SetArg(const cl_uint index, const VectorFloat3DType & value);

  using VectorDouble3DType = Vector<double, 3>;
  cl_int
  SetArg(const cl_uint index, const VectorDouble3DType & value);

  using VectorInt4DType = Vector<int, 4>;
  cl_int
  SetArg(const cl_uint index, const VectorInt4DType & value);

  using VectorFloat4DType = Vector<float, 4>;
  cl_int
  SetArg(const cl_uint index, const VectorFloat4DType & value);

  using VectorDouble4DType = Vector<double, 4>;
  cl_int
  SetArg(const cl_uint index, const VectorDouble4DType & value);

  /** Sets argument \a index for this kernel to covariant vector \a value. */
  using CovariantVectorInt1DType = CovariantVector<int, 1>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorInt1DType & value);

  using CovariantVectorFloat1DType = CovariantVector<float, 1>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorFloat1DType & value);

  using CovariantVectorDouble1DType = CovariantVector<double, 1>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorDouble1DType & value);

  using CovariantVectorInt2DType = CovariantVector<int, 2>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorInt2DType & value);

  using CovariantVectorFloat2DType = CovariantVector<float, 2>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorFloat2DType & value);

  using CovariantVectorDouble2DType = CovariantVector<double, 2>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorDouble2DType & value);

  using CovariantVectorInt3DType = CovariantVector<int, 3>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorInt3DType & value);

  using CovariantVectorFloat3DType = CovariantVector<float, 3>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorFloat3DType & value);

  using CovariantVectorDouble3DType = CovariantVector<double, 3>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorDouble3DType & value);

  using CovariantVectorInt4DType = CovariantVector<int, 4>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorInt4DType & value);

  using CovariantVectorFloat4DType = CovariantVector<float, 4>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorFloat4DType & value);

  using CovariantVectorDouble4DType = CovariantVector<double, 4>;
  cl_int
  SetArg(const cl_uint index, const CovariantVectorDouble4DType & value);

  /** Sets argument \a index for this kernel to matrix \a value. */
  using MatrixFloat1x1Type = Matrix<float, 1, 1>;
  cl_int
  SetArg(const cl_uint index, const MatrixFloat1x1Type & value);

  using MatrixDouble1x1Type = Matrix<double, 1, 1>;
  cl_int
  SetArg(const cl_uint index, const MatrixDouble1x1Type & value);

  using MatrixFloat2x2Type = Matrix<float, 2, 2>;
  cl_int
  SetArg(const cl_uint index, const MatrixFloat2x2Type & value);

  using MatrixDouble2x2Type = Matrix<double, 2, 2>;
  cl_int
  SetArg(const cl_uint index, const MatrixDouble2x2Type & value);

  using MatrixFloat3x3Type = Matrix<float, 3, 3>;
  cl_int
  SetArg(const cl_uint index, const MatrixFloat3x3Type & value);

  using MatrixDouble3x3Type = Matrix<double, 3, 3>;
  cl_int
  SetArg(const cl_uint index, const MatrixDouble3x3Type & value);

  using MatrixFloat4x4Type = Matrix<float, 4, 4>;
  cl_int
  SetArg(const cl_uint index, const MatrixFloat4x4Type & value);

  using MatrixDouble4x4Type = Matrix<double, 4, 4>;
  cl_int
  SetArg(const cl_uint index, const MatrixDouble4x4Type & value);

  /** Sets argument \a index for this kernel to \a value.
   * The argument is assumed to have been declared with the
   * type \c image1d_t, \c image2d_t, \c image3d_t, or be a pointer to a buffer,
   * according to the type of memory object represented by \a value. */
  cl_int
  SetArg(const cl_uint index, const OpenCLMemoryObject & value);

  /** Sets argument \a index for this kernel to \a value.
   * The argument is assumed to have been declared as a pointer
   * to a buffer. */
  cl_int
  SetArg(const cl_uint index, const OpenCLVectorBase & value);

  /** Sets argument \a index for this kernel to \a value.
   * The argument is assumed to have been declared with the type \c sampler_t. */
  cl_int
  SetArg(const cl_uint index, const OpenCLSampler & value);

  /** Requests that this kernel instance be run on GetGlobalWorkSize() items,
   * optionally subdivided into work groups of GetLocalWorkSize() items.
   * Returns an event object that can be used to wait for the kernel
   * to finish execution. The request is executed on the active
   * command queue for GetContext().
   * \sa operator()() */
  OpenCLEvent
  LaunchKernel();

  OpenCLEvent
  LaunchKernel(const OpenCLSize & global_work_size,
               const OpenCLSize & local_work_size = OpenCLSize::null,
               const OpenCLSize & global_work_offset = OpenCLSize::null);

  /** Requests that this kernel instance be run on GetGlobalWorkSize() items,
   * optionally subdivided into work groups of GetLocalWorkSize() items.
   * If \a event_list is not an empty list, it indicates the events that must
   * be signaled as finished before this kernel instance can begin executing.
   * Returns an event object that can be used to wait for the kernel
   * to finish execution. The request is executed on the active
   * command queue for GetContext(). */
  OpenCLEvent
  LaunchKernel(const OpenCLEventList & event_list);

  OpenCLEvent
  LaunchKernel(const OpenCLEventList & event_list,
               const OpenCLSize &      global_work_size,
               const OpenCLSize &      local_work_size = OpenCLSize::null,
               const OpenCLSize &      global_work_offset = OpenCLSize::null);

  /** Enqueues a command to execute a kernel on a device. The kernel is executed
   * using a single work-item. Returns true if the task was successful,
   * false otherwise. This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa LaunchTaskAsync() */
  bool
  LaunchTask(const OpenCLEventList & event_list);

  /** Asynchronous version of the LaunchTask() method.
   * This function will queue the task \a event_list and return immediately.
   * Returns an OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa LaunchTask(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  LaunchTaskAsync(const OpenCLEventList & event_list);

  /** Runs this kernel instance with zero arguments.
   * Returns an event object that can be used to wait for the
   * kernel to finish execution. */
  inline OpenCLEvent
  operator()()
  {
    return this->LaunchKernel();
  }

  /** Runs this kernel instance with the argument \a arg1.
   * Returns an event object that can be used to wait for the
   * kernel to finish execution. */
  template <typename T1>
  inline OpenCLEvent
  operator()(const T1 & arg1)
  {
    this->SetArg(0, arg1);
    return this->LaunchKernel();
  }


  /** Runs this kernel instance with the arguments \a arg1 and \a arg2.
   * Returns an event object that can be used to wait for the
   * kernel to finish execution. */
  template <typename T1, typename T2>
  inline OpenCLEvent
  operator()(const T1 & arg1, const T2 & arg2)
  {
    this->SetArg(0, arg1);
    this->SetArg(1, arg2);
    return this->LaunchKernel();
  }


  /** Runs this kernel instance with the arguments \a arg1, \a arg2,
   * and \a arg3. Returns an event object that can be used to wait for the
   * kernel to finish execution. */
  template <typename T1, typename T2, typename T3>
  inline OpenCLEvent
  operator()(const T1 & arg1, const T2 & arg2, const T3 & arg3)
  {
    this->SetArg(0, arg1);
    this->SetArg(1, arg2);
    this->SetArg(2, arg3);
    return this->LaunchKernel();
  }


  /** Runs this kernel instance with the arguments \a arg1, \a arg2,
   * \a arg3, and \a arg4. Returns an event object that can be used to
   * wait for the kernel to finish execution. */
  template <typename T1, typename T2, typename T3, typename T4>
  inline OpenCLEvent
  operator()(const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4)
  {
    this->SetArg(0, arg1);
    this->SetArg(1, arg2);
    this->SetArg(2, arg3);
    this->SetArg(3, arg4);
    return this->LaunchKernel();
  }


  /** Runs this kernel instance with the arguments \a arg1, \a arg2,
   * \a arg3, \a arg4, and \a arg5. Returns an event object that can be
   * used to wait for the kernel to finish execution. */
  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  inline OpenCLEvent
  operator()(const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4, const T5 & arg5)
  {
    this->SetArg(0, arg1);
    this->SetArg(1, arg2);
    this->SetArg(2, arg3);
    this->SetArg(3, arg4);
    this->SetArg(4, arg5);
    return this->LaunchKernel();
  }


  /** Runs this kernel instance with the arguments \a arg1, \a arg2,
   * \a arg3, \a arg4, \a arg5, and \a arg6. Returns an event object that
   * can be used to wait for the kernel to finish execution. */
  template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
  inline OpenCLEvent
  operator()(const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4, const T5 & arg5, const T6 & arg6)
  {
    this->SetArg(0, arg1);
    this->SetArg(1, arg2);
    this->SetArg(2, arg3);
    this->SetArg(3, arg4);
    this->SetArg(4, arg5);
    this->SetArg(5, arg6);
    return this->LaunchKernel();
  }


  /** Runs this kernel instance with the arguments \a arg1, \a arg2,
   * \a arg3, \a arg4, \a arg5, \a arg6, and \a arg7. Returns an event
   * object that can be used to wait for the kernel to finish execution. */
  template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
  inline OpenCLEvent
  operator()(const T1 & arg1,
             const T2 & arg2,
             const T3 & arg3,
             const T4 & arg4,
             const T5 & arg5,
             const T6 & arg6,
             const T7 & arg7)
  {
    this->SetArg(0, arg1);
    this->SetArg(1, arg2);
    this->SetArg(2, arg3);
    this->SetArg(3, arg4);
    this->SetArg(4, arg5);
    this->SetArg(5, arg6);
    this->SetArg(6, arg7);
    return this->LaunchKernel();
  }


  /** Runs this kernel instance with the arguments \a arg1, \a arg2,
   * \a arg3, \a arg4, \a arg5, \a arg6, \a arg7, and \a arg8. Returns
   * an event object that can be used to wait for the kernel to finish execution. */
  template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
  inline OpenCLEvent
  operator()(const T1 & arg1,
             const T2 & arg2,
             const T3 & arg3,
             const T4 & arg4,
             const T5 & arg5,
             const T6 & arg6,
             const T7 & arg7,
             const T8 & arg8)
  {
    this->SetArg(0, arg1);
    this->SetArg(1, arg2);
    this->SetArg(2, arg3);
    this->SetArg(3, arg4);
    this->SetArg(4, arg5);
    this->SetArg(5, arg6);
    this->SetArg(6, arg7);
    this->SetArg(7, arg8);
    return this->LaunchKernel();
  }


  /** Runs this kernel instance with the arguments \a arg1, \a arg2,
   * \a arg3, \a arg4, \a arg5, \a arg6, \a arg7, \a arg8, and \a arg9.
   * Returns an event object that can be used to wait for the kernel
   * to finish execution. */
  template <typename T1,
            typename T2,
            typename T3,
            typename T4,
            typename T5,
            typename T6,
            typename T7,
            typename T8,
            typename T9>
  inline OpenCLEvent
  operator()(const T1 & arg1,
             const T2 & arg2,
             const T3 & arg3,
             const T4 & arg4,
             const T5 & arg5,
             const T6 & arg6,
             const T7 & arg7,
             const T8 & arg8,
             const T9 & arg9)
  {
    this->SetArg(0, arg1);
    this->SetArg(1, arg2);
    this->SetArg(2, arg3);
    this->SetArg(3, arg4);
    this->SetArg(4, arg5);
    this->SetArg(5, arg6);
    this->SetArg(6, arg7);
    this->SetArg(7, arg8);
    this->SetArg(8, arg9);
    return this->LaunchKernel();
  }


  /** Runs this kernel instance with the arguments \a arg1, \a arg2,
   * \a arg3, \a arg4, \a arg5, \a arg6, \a arg7, \a arg8, \a arg9,
   * and \a arg10. Returns an event object that can be used to wait
   * for the kernel to finish execution. */
  template <typename T1,
            typename T2,
            typename T3,
            typename T4,
            typename T5,
            typename T6,
            typename T7,
            typename T8,
            typename T9,
            typename T10>
  inline OpenCLEvent
  operator()(const T1 &  arg1,
             const T2 &  arg2,
             const T3 &  arg3,
             const T4 &  arg4,
             const T5 &  arg5,
             const T6 &  arg6,
             const T7 &  arg7,
             const T8 &  arg8,
             const T9 &  arg9,
             const T10 & arg10)
  {
    this->SetArg(0, arg1);
    this->SetArg(1, arg2);
    this->SetArg(2, arg3);
    this->SetArg(3, arg4);
    this->SetArg(4, arg5);
    this->SetArg(5, arg6);
    this->SetArg(6, arg7);
    this->SetArg(7, arg8);
    this->SetArg(8, arg9);
    this->SetArg(9, arg10);
    return this->LaunchKernel();
  }


private:
  std::unique_ptr<OpenCLKernelPimpl> d_ptr;
  cl_kernel                          m_KernelId;
  bool                               m_DoubleAsFloat;

  ITK_OPENCL_DECLARE_PRIVATE(OpenCLKernel)
};

/** Operator ==
 * Returns true if \a lhs OpenCL kernel identifier is the same as \a rhs, false otherwise.
 * \sa operator!=, operator==(), GetKernelId() */
bool ITKOpenCL_EXPORT
     operator==(const OpenCLKernel & lhs, const OpenCLKernel & rhs);

/** Operator !=
 * Returns true if \a lhs OpenCL kernel identifier is not the same as \a rhs, false otherwise.
 * \sa operator==, operator==(), GetKernelId() */
bool ITKOpenCL_EXPORT
     operator!=(const OpenCLKernel & lhs, const OpenCLKernel & rhs);

/** Stream out operator for OpenCLKernel */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLKernel & kernel)
{
  if (kernel.IsNull())
  {
    strm << "OpenCLKernel(null)";
    return strm;
  }

  const char indent = ' ';

  strm << "OpenCLKernel\n"
       << indent << "Id: " << kernel.GetKernelId() << '\n'
       << indent << "Name: " << kernel.GetName() << '\n'
       << indent << "Number of arguments: " << kernel.GetNumberOfArguments() << '\n'
       << indent << "Global work size: " << kernel.GetGlobalWorkSize() << '\n'
       << indent << "Local work size: " << kernel.GetLocalWorkSize() << '\n'
       << indent << "Global work offset: " << kernel.GetGlobalWorkOffset() << '\n'
       << indent << "Compile work group size: " << kernel.GetCompileWorkGroupSize() << '\n'
       << indent << "Best local work size image 2D: " << kernel.GetBestLocalWorkSizeImage2D() << '\n'
       << indent << "Best local work size image 3D: " << kernel.GetBestLocalWorkSizeImage3D() << '\n'
       << indent << "Preferred work size multiple: " << kernel.GetPreferredWorkSizeMultiple() << std::endl;

  return strm;
}


} // end namespace itk

#endif /* itkOpenCLKernel_h */
