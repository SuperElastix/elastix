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
#include "itkOpenCLKernel.h"
#include "itkOpenCLProgram.h"
#include "itkOpenCLBuffer.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLExtension.h"
#include "itkOpenCLMacro.h"

namespace itk
{
class OpenCLKernelPimpl
{
public:

  OpenCLKernelPimpl( OpenCLContext * ctx, const cl_kernel kid ) :
    context( ctx ),
    id( kid ),
    global_work_offset( OpenCLSize::null ),
    global_work_size( 1 ),
    local_work_size( OpenCLSize::null )
  {}
  OpenCLKernelPimpl( const OpenCLKernelPimpl * other ) :
    context( other->context ),
    id( other->id ),
    global_work_offset( other->global_work_offset ),
    global_work_size( other->global_work_size ),
    local_work_size( other->local_work_size )
  {
    if( id )
    {
      clRetainKernel( id );
    }
  }


  ~OpenCLKernelPimpl()
  {
    if( id )
    {
      clReleaseKernel( id );
    }
  }


  void copy( const OpenCLKernelPimpl * other )
  {
    context = other->context;

    global_work_offset = other->global_work_offset;
    global_work_size   = other->global_work_size;
    local_work_size    = other->local_work_size;

    if( id != other->id )
    {
      if( id )
      {
        clReleaseKernel( id );
      }
      id = other->id;
      if( id )
      {
        clRetainKernel( id );
      }
    }
  }


  OpenCLContext * context;
  cl_kernel       id;
  OpenCLSize      global_work_offset;
  OpenCLSize      global_work_size;
  OpenCLSize      local_work_size;
};

//------------------------------------------------------------------------------
// Implementations of SetArg() methods for types:
// char(n), uchar(n), short(n), ushort(n), int(n), uint(n),
// long(n), ulong(n), float(n), double(n).
OpenCLKernelSetArgsMacroCXX( cl_uchar, cl_uchar2, cl_uchar4, cl_uchar8, cl_uchar16 )
OpenCLKernelSetArgsMacroCXX( cl_char, cl_char2, cl_char4, cl_char8, cl_char16 )
OpenCLKernelSetArgsMacroCXX( cl_ushort, cl_ushort2, cl_ushort4, cl_ushort8, cl_ushort16 )
OpenCLKernelSetArgsMacroCXX( cl_short, cl_short2, cl_short4, cl_short8, cl_short16 )
OpenCLKernelSetArgsMacroCXX( cl_uint, cl_uint2, cl_uint4, cl_uint8, cl_uint16 )
OpenCLKernelSetArgsMacroCXX( cl_int, cl_int2, cl_int4, cl_int8, cl_int16 )
OpenCLKernelSetArgsMacroCXX( cl_ulong, cl_ulong2, cl_ulong4, cl_ulong8, cl_ulong16 )
OpenCLKernelSetArgsMacroCXX( cl_long, cl_long2, cl_long4, cl_long8, cl_long16 )
OpenCLKernelSetArgsMacroCXX( cl_float, cl_float2, cl_float4, cl_float8, cl_float16 )
OpenCLKernelSetArgsMacroCXX( cl_double, cl_double2, cl_double4, cl_double8, cl_double16 )

//------------------------------------------------------------------------------
OpenCLKernel::OpenCLKernel() :
  d_ptr( new OpenCLKernelPimpl( 0, 0 ) ), m_KernelId( 0 ), m_DoubleAsFloat( true )
{}

//------------------------------------------------------------------------------
OpenCLKernel::OpenCLKernel( OpenCLContext * context, const cl_kernel id ) :
  d_ptr( new OpenCLKernelPimpl( context, id ) ), m_KernelId( id ), m_DoubleAsFloat( true )
{}

//------------------------------------------------------------------------------
OpenCLKernel::OpenCLKernel( const OpenCLKernel & other ) :
  d_ptr( new OpenCLKernelPimpl( other.d_ptr.get() ) ),
  m_KernelId( other.m_KernelId ),
  m_DoubleAsFloat( true )
{}

//------------------------------------------------------------------------------
// Destructor has to be in cxx, otherwise compiler will print warning messages.
OpenCLKernel::~OpenCLKernel()
{}

//------------------------------------------------------------------------------
OpenCLKernel &
OpenCLKernel::operator=( const OpenCLKernel & other )
{
  this->d_ptr->copy( other.d_ptr.get() );
  this->m_KernelId      = other.m_KernelId;
  this->m_DoubleAsFloat = other.m_DoubleAsFloat;
  return *this;
}


//------------------------------------------------------------------------------
bool
OpenCLKernel::IsNull() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  return d->id == 0;
}


//------------------------------------------------------------------------------
cl_kernel
OpenCLKernel::GetKernelId() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  return d->id;
}


//------------------------------------------------------------------------------
OpenCLContext *
OpenCLKernel::GetContext() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  return d->context;
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLKernel::GetProgram() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  if( this->IsNull() )
  {
    return OpenCLProgram();
  }
  cl_program prog = 0;
  if( clGetKernelInfo( d->id, CL_KERNEL_PROGRAM,
    sizeof( prog ), &prog, 0 ) != CL_SUCCESS )
  {
    return OpenCLProgram();
  }
  return OpenCLProgram( d->context, prog );
}


//------------------------------------------------------------------------------
std::string
OpenCLKernel::GetName() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  size_t size = 0;
  if( clGetKernelInfo( d->id, CL_KERNEL_FUNCTION_NAME,
    0, 0, &size ) != CL_SUCCESS || !size )
  {
    return std::string();
  }
  std::string buffer( size, '\0' );
  if( clGetKernelInfo( d->id, CL_KERNEL_FUNCTION_NAME, size, &buffer[ 0 ], 0 ) != CL_SUCCESS )
  {
    return std::string();
  }
  return buffer;
}


//------------------------------------------------------------------------------
std::size_t
OpenCLKernel::GetNumberOfArguments() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  cl_uint count = 0;
  if( clGetKernelInfo( d->id, CL_KERNEL_NUM_ARGS, sizeof( count ), &count, 0 )
    != CL_SUCCESS )
  {
    return 0;
  }
  return std::size_t( count );
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLKernel::GetCompileWorkGroupSize() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  size_t sizes[ 3 ];
  if( clGetKernelWorkGroupInfo
      ( d->id, d->context->GetDefaultDevice().GetDeviceId(),
    CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
    sizeof( sizes ), sizes, 0 ) != CL_SUCCESS )
  {
    return OpenCLSize( 0, 0, 0 );
  }
  else
  {
    return OpenCLSize( sizes[ 0 ], sizes[ 1 ], sizes[ 2 ] );
  }
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLKernel::GetCompileWorkGroupSize( const OpenCLDevice & device ) const
{
  ITK_OPENCL_D( const OpenCLKernel );
  size_t sizes[ 3 ];
  if( clGetKernelWorkGroupInfo
      ( d->id, device.GetDeviceId(),
    CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
    sizeof( sizes ), sizes, 0 ) != CL_SUCCESS )
  {
    return OpenCLSize( 0, 0, 0 );
  }
  else
  {
    return OpenCLSize( sizes[ 0 ], sizes[ 1 ], sizes[ 2 ] );
  }
}


//------------------------------------------------------------------------------
void
OpenCLKernel::SetGlobalWorkSize( const OpenCLSize & size )
{
  ITK_OPENCL_D( OpenCLKernel );
  d->global_work_size = size;
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLKernel::GetGlobalWorkSize() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  return d->global_work_size;
}


//------------------------------------------------------------------------------
void
OpenCLKernel::SetRoundedGlobalWorkSize( const OpenCLSize & size )
{
  this->SetGlobalWorkSize( size.RoundTo( GetLocalWorkSize() ) );
}


//------------------------------------------------------------------------------
void
OpenCLKernel::SetLocalWorkSize( const OpenCLSize & size )
{
  ITK_OPENCL_D( OpenCLKernel );
  d->local_work_size = size;
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLKernel::GetLocalWorkSize() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  return d->local_work_size;
}


//------------------------------------------------------------------------------
void
OpenCLKernel::SetGlobalWorkOffset( const OpenCLSize & offset )
{
  ITK_OPENCL_D( OpenCLKernel );
  d->global_work_offset = offset;
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLKernel::GetGlobalWorkOffset() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  return d->global_work_offset;
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLKernel::GetBestLocalWorkSizeImage1D() const
{
  const std::list< OpenCLDevice > devices  = this->GetProgram().GetDevices();
  const size_t                    maxItems = devices.empty() ? 1 : devices.front().GetMaximumWorkItemsPerGroup();
  size_t                          size     = 8;

  while( size > 1 && ( size ) > maxItems )
  {
    size /= 2;
  }
  return OpenCLSize( size );
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLKernel::GetBestLocalWorkSizeImage2D() const
{
  const std::list< OpenCLDevice > devices  = this->GetProgram().GetDevices();
  const size_t                    maxItems = devices.empty() ? 1 : devices.front().GetMaximumWorkItemsPerGroup();
  size_t                          size     = 8;

  while( size > 1 && ( size * size ) > maxItems )
  {
    size /= 2;
  }
  return OpenCLSize( size, size );
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLKernel::GetBestLocalWorkSizeImage3D() const
{
  const std::list< OpenCLDevice > devices  = this->GetProgram().GetDevices();
  const size_t                    maxItems = devices.empty() ? 1 : devices.front().GetMaximumWorkItemsPerGroup();
  size_t                          size     = 8;

  while( size > 1 && ( size * size * size ) > maxItems )
  {
    size /= 2;
  }
  return OpenCLSize( size, size, size );
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLKernel::GetBestLocalWorkSizeImage( const std::size_t dimension ) const
{
  switch( dimension )
  {
    case 1:
      return this->GetBestLocalWorkSizeImage1D();
      break;
    case 2:
      return this->GetBestLocalWorkSizeImage2D();
      break;
    case 3:
      return this->GetBestLocalWorkSizeImage3D();
      break;
    default:
      itkOpenCLErrorMacroGeneric( << "Not supported dimension." );
      return OpenCLSize();
      break;
  }
}


//------------------------------------------------------------------------------
size_t
OpenCLKernel::GetPreferredWorkSizeMultiple() const
{
  ITK_OPENCL_D( const OpenCLKernel );
  size_t size;
  if( clGetKernelWorkGroupInfo
      ( d->id, d->context->GetDefaultDevice().GetDeviceId(),
    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
    sizeof( size ), &size, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return size;
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const void * data, const size_t size )
{
  return clSetKernelArg( this->m_KernelId, index, size, data );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Size1DType & value )
{
  const cl_uint values = value[ 0 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Size2DType & value )
{
  cl_uint2 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Size3DType & value )
{
  cl_uint3 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Size4DType & value )
{
  cl_uint4 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  values.s[ 3 ] = value[ 3 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Index1DType & value )
{
  const cl_int values = value[ 0 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Index2DType & value )
{
  cl_int2 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Index3DType & value )
{
  cl_int3 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Index4DType & value )
{
  cl_int4 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  values.s[ 3 ] = value[ 3 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Offset1DType & value )
{
  const cl_int values = value[ 0 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Offset2DType & value )
{
  cl_int2 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Offset3DType & value )
{
  cl_int3 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const Offset4DType & value )
{
  cl_int4 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  values.s[ 3 ] = value[ 3 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointInt1DType & value )
{
  const cl_int values = value[ 0 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointFloat1DType & value )
{
  const cl_float values = value[ 0 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointDouble1DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    const cl_double values = value[ 0 ];
    return this->SetArg( index, values );
  }
  else
  {
    const cl_float values = static_cast< float >( value[ 0 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointInt2DType & value )
{
  cl_int2 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointFloat2DType & value )
{
  cl_float2 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointDouble2DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    cl_double2 values;
    values.s[ 0 ] = value[ 0 ];
    values.s[ 1 ] = value[ 1 ];
    return this->SetArg( index, values );
  }
  else
  {
    cl_float2 values;
    values.s[ 0 ] = static_cast< float >( value[ 0 ] );
    values.s[ 1 ] = static_cast< float >( value[ 1 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointInt3DType & value )
{
  cl_int3 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointFloat3DType & value )
{
  cl_float3 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointDouble3DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    cl_double3 values;
    values.s[ 0 ] = value[ 0 ];
    values.s[ 1 ] = value[ 1 ];
    values.s[ 2 ] = value[ 2 ];
    return this->SetArg( index, values );
  }
  else
  {
    cl_float3 values;
    values.s[ 0 ] = static_cast< float >( value[ 0 ] );
    values.s[ 1 ] = static_cast< float >( value[ 1 ] );
    values.s[ 2 ] = static_cast< float >( value[ 2 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointInt4DType & value )
{
  cl_int4 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  values.s[ 3 ] = value[ 3 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointFloat4DType & value )
{
  cl_float4 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  values.s[ 3 ] = value[ 3 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const PointDouble4DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    cl_double4 values;
    values.s[ 0 ] = value[ 0 ];
    values.s[ 1 ] = value[ 1 ];
    values.s[ 2 ] = value[ 2 ];
    values.s[ 3 ] = value[ 3 ];
    return this->SetArg( index, values );
  }
  else
  {
    cl_float4 values;
    values.s[ 0 ] = static_cast< float >( value[ 0 ] );
    values.s[ 1 ] = static_cast< float >( value[ 1 ] );
    values.s[ 2 ] = static_cast< float >( value[ 2 ] );
    values.s[ 3 ] = static_cast< float >( value[ 3 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorInt1DType & value )
{
  const cl_int values = value[ 0 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorFloat1DType & value )
{
  const cl_float values = value[ 0 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorDouble1DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    const cl_double values = value[ 0 ];
    return this->SetArg( index, values );
  }
  else
  {
    const cl_float values = static_cast< float >( value[ 0 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorInt2DType & value )
{
  cl_int2 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorFloat2DType & value )
{
  cl_float2 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorDouble2DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    cl_double2 values;
    values.s[ 0 ] = value[ 0 ];
    values.s[ 1 ] = value[ 1 ];
    return this->SetArg( index, values );
  }
  else
  {
    cl_float2 values;
    values.s[ 0 ] = static_cast< float >( value[ 0 ] );
    values.s[ 1 ] = static_cast< float >( value[ 1 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorInt3DType & value )
{
  cl_int3 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorFloat3DType & value )
{
  cl_float3 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorDouble3DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    cl_double3 values;
    values.s[ 0 ] = value[ 0 ];
    values.s[ 1 ] = value[ 1 ];
    values.s[ 2 ] = value[ 2 ];
    return this->SetArg( index, values );
  }
  else
  {
    cl_float3 values;
    values.s[ 0 ] = static_cast< float >( value[ 0 ] );
    values.s[ 1 ] = static_cast< float >( value[ 1 ] );
    values.s[ 2 ] = static_cast< float >( value[ 2 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorInt4DType & value )
{
  cl_int4 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  values.s[ 3 ] = value[ 3 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorFloat4DType & value )
{
  cl_float4 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  values.s[ 3 ] = value[ 3 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const VectorDouble4DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    cl_double4 values;
    values.s[ 0 ] = value[ 0 ];
    values.s[ 1 ] = value[ 1 ];
    values.s[ 2 ] = value[ 2 ];
    values.s[ 3 ] = value[ 3 ];
    return this->SetArg( index, values );
  }
  else
  {
    cl_float4 values;
    values.s[ 0 ] = static_cast< float >( value[ 0 ] );
    values.s[ 1 ] = static_cast< float >( value[ 1 ] );
    values.s[ 2 ] = static_cast< float >( value[ 2 ] );
    values.s[ 3 ] = static_cast< float >( value[ 3 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorInt1DType & value )
{
  const cl_int values = value[ 0 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorFloat1DType & value )
{
  const cl_float values = value[ 0 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorDouble1DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    const cl_double values = value[ 0 ];
    return this->SetArg( index, values );
  }
  else
  {
    const cl_float values = static_cast< float >( value[ 0 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorInt2DType & value )
{
  cl_int2 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorFloat2DType & value )
{
  cl_float2 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorDouble2DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    cl_double2 values;
    values.s[ 0 ] = value[ 0 ];
    values.s[ 1 ] = value[ 1 ];
    return this->SetArg( index, values );
  }
  else
  {
    cl_float2 values;
    values.s[ 0 ] = static_cast< float >( value[ 0 ] );
    values.s[ 1 ] = static_cast< float >( value[ 1 ] );
    return this->SetArg( index, values );
  }

}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorInt3DType & value )
{
  cl_int3 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorFloat3DType & value )
{
  cl_float3 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorDouble3DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    cl_double3 values;
    values.s[ 0 ] = value[ 0 ];
    values.s[ 1 ] = value[ 1 ];
    values.s[ 2 ] = value[ 2 ];
    return this->SetArg( index, values );
  }
  else
  {
    cl_float3 values;
    values.s[ 0 ] = static_cast< float >( value[ 0 ] );
    values.s[ 1 ] = static_cast< float >( value[ 1 ] );
    values.s[ 2 ] = static_cast< float >( value[ 2 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorInt4DType & value )
{
  cl_int4 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  values.s[ 3 ] = value[ 3 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorFloat4DType & value )
{
  cl_float4 values;
  values.s[ 0 ] = value[ 0 ];
  values.s[ 1 ] = value[ 1 ];
  values.s[ 2 ] = value[ 2 ];
  values.s[ 3 ] = value[ 3 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const CovariantVectorDouble4DType & value )
{
  if( !this->m_DoubleAsFloat )
  {
    cl_double4 values;
    values.s[ 0 ] = value[ 0 ];
    values.s[ 1 ] = value[ 1 ];
    values.s[ 2 ] = value[ 2 ];
    values.s[ 3 ] = value[ 3 ];
    return this->SetArg( index, values );
  }
  else
  {
    cl_float4 values;
    values.s[ 0 ] = static_cast< float >( value[ 0 ] );
    values.s[ 1 ] = static_cast< float >( value[ 1 ] );
    values.s[ 2 ] = static_cast< float >( value[ 2 ] );
    values.s[ 3 ] = static_cast< float >( value[ 3 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const MatrixFloat1x1Type & value )
{
  const cl_float values = value[ 0 ][ 0 ];
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const MatrixDouble1x1Type & value )
{
  if( !this->m_DoubleAsFloat )
  {
    const cl_double values = value[ 0 ][ 0 ];
    return this->SetArg( index, values );
  }
  else
  {
    const cl_float values = static_cast< float >( value[ 0 ][ 0 ] );
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const MatrixFloat2x2Type & value )
{
  cl_float4          values;
  unsigned int       id       = 0;
  const unsigned int nRows    = MatrixFloat2x2Type::RowDimensions;
  const unsigned int nColumns = MatrixFloat2x2Type::ColumnDimensions;

  for( unsigned int i = 0; i < nRows; i++ )
  {
    for( unsigned int j = 0; j < nColumns; j++ )
    {
      values.s[ id ] = value[ i ][ j ];
      id++;
    }
  }
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const MatrixDouble2x2Type & value )
{
  unsigned int       id       = 0;
  const unsigned int nRows    = MatrixDouble2x2Type::RowDimensions;
  const unsigned int nColumns = MatrixDouble2x2Type::ColumnDimensions;

  if( !this->m_DoubleAsFloat )
  {
    cl_double4 values;
    for( unsigned int i = 0; i < nRows; i++ )
    {
      for( unsigned int j = 0; j < nColumns; j++ )
      {
        values.s[ id ] = value[ i ][ j ];
        id++;
      }
    }
    return this->SetArg( index, values );
  }
  else
  {
    cl_float4 values;
    for( unsigned int i = 0; i < nRows; i++ )
    {
      for( unsigned int j = 0; j < nColumns; j++ )
      {
        values.s[ id ] = static_cast< float >( value[ i ][ j ] );
        id++;
      }
    }
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const MatrixFloat3x3Type & value )
{
  // OpenCL does not support float9 therefore we are using float16
  cl_float16         values;
  unsigned int       id       = 0;
  const unsigned int nRows    = MatrixFloat3x3Type::RowDimensions;
  const unsigned int nColumns = MatrixFloat3x3Type::ColumnDimensions;

  for( unsigned int i = 0; i < nRows; i++ )
  {
    for( unsigned int j = 0; j < nColumns; j++ )
    {
      values.s[ id ] = value[ i ][ j ];
      id++;
    }
  }
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const MatrixDouble3x3Type & value )
{
  unsigned int       id       = 0;
  const unsigned int nRows    = MatrixDouble3x3Type::RowDimensions;
  const unsigned int nColumns = MatrixDouble3x3Type::ColumnDimensions;

  if( !this->m_DoubleAsFloat )
  {
    // OpenCL does not support double9 therefore we are using double16
    cl_double16 values;
    for( unsigned int i = 0; i < nRows; i++ )
    {
      for( unsigned int j = 0; j < nColumns; j++ )
      {
        values.s[ id ] = value[ i ][ j ];
        id++;
      }
    }
    return this->SetArg( index, values );
  }
  else
  {
    // OpenCL does not support float9 therefore we are using float16
    cl_float16 values;
    for( unsigned int i = 0; i < nRows; i++ )
    {
      for( unsigned int j = 0; j < nColumns; j++ )
      {
        values.s[ id ] = static_cast< float >( value[ i ][ j ] );
        id++;
      }
    }
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const MatrixFloat4x4Type & value )
{
  // OpenCL does not support float9 therefore we are using float16
  cl_float16         values;
  unsigned int       id       = 0;
  const unsigned int nRows    = MatrixFloat4x4Type::RowDimensions;
  const unsigned int nColumns = MatrixFloat4x4Type::ColumnDimensions;

  for( unsigned int i = 0; i < nRows; i++ )
  {
    for( unsigned int j = 0; j < nColumns; j++ )
    {
      values.s[ id ] = value[ i ][ j ];
      id++;
    }
  }
  return this->SetArg( index, values );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const MatrixDouble4x4Type & value )
{
  unsigned int       id       = 0;
  const unsigned int nRows    = MatrixDouble4x4Type::RowDimensions;
  const unsigned int nColumns = MatrixDouble4x4Type::ColumnDimensions;

  if( !this->m_DoubleAsFloat )
  {
    // OpenCL does not support double9 therefore we are using double16
    cl_double16 values;
    for( unsigned int i = 0; i < nRows; i++ )
    {
      for( unsigned int j = 0; j < nColumns; j++ )
      {
        values.s[ id ] = value[ i ][ j ];
        id++;
      }
    }
    return this->SetArg( index, values );
  }
  else
  {
    // OpenCL does not support float9 therefore we are using float16
    cl_float16 values;
    for( unsigned int i = 0; i < nRows; i++ )
    {
      for( unsigned int j = 0; j < nColumns; j++ )
      {
        values.s[ id ] = static_cast< float >( value[ i ][ j ] );
        id++;
      }
    }
    return this->SetArg( index, values );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const OpenCLMemoryObject & value )
{
  cl_mem id = value.GetMemoryId();

  return clSetKernelArg( this->m_KernelId, index, sizeof( id ), &id );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const OpenCLVectorBase & value )
{
  cl_mem id = value.GetKernelArgument();

  return clSetKernelArg( this->m_KernelId, index, sizeof( id ), &id );
}


//------------------------------------------------------------------------------
cl_int
OpenCLKernel::SetArg( const cl_uint index, const OpenCLSampler & value )
{
  cl_sampler id = value.GetSamplerId();

  return clSetKernelArg( this->m_KernelId, index, sizeof( id ), &id );
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLKernel::LaunchKernel()
{
  ITK_OPENCL_D( const OpenCLKernel );

  const cl_uint work_dim = d->global_work_size.GetDimension();
  const bool    gwoNull  = d->global_work_offset.IsNull();
  const bool    lwsNull  = d->local_work_size.IsNull();

  cl_event event;
  cl_int   error;

  if( gwoNull && lwsNull )
  {
    error = clEnqueueNDRangeKernel( d->context->GetActiveQueue(), this->m_KernelId,
      work_dim, NULL, d->global_work_size.GetSizes(), NULL, 0, 0, &event );
  }
  else if( gwoNull && !lwsNull )
  {
    error = clEnqueueNDRangeKernel( d->context->GetActiveQueue(), this->m_KernelId,
      work_dim, NULL, d->global_work_size.GetSizes(),
      ( d->local_work_size.GetWidth() ? d->local_work_size.GetSizes() : 0 ),

      0, 0, &event );
  }
  else if( !gwoNull && lwsNull )
  {
    error = clEnqueueNDRangeKernel( d->context->GetActiveQueue(), this->m_KernelId,
      work_dim, d->global_work_offset.GetSizes(), d->global_work_size.GetSizes(),
      NULL, 0, 0, &event );
  }
  else
  {
    error = clEnqueueNDRangeKernel( d->context->GetActiveQueue(), this->m_KernelId,
      work_dim, d->global_work_offset.GetSizes(), d->global_work_size.GetSizes(),
      ( d->local_work_size.GetWidth() ? d->local_work_size.GetSizes() : 0 ),
      0, 0, &event );
  }

  if( error != CL_SUCCESS )
  {
    //const std::size_t num = GetNumberOfArguments();
    itkOpenCLErrorMacroGeneric( << "Launch kernel '" << this->GetName() << "' failed." );
    d->context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
    return OpenCLEvent();
  }
  else
  {
#ifdef OPENCL_PROFILING
    const std::string profileStr = "clEnqueueNDRangeKernel: " + this->GetName();
    d->context->OpenCLProfile( event, profileStr );
#endif
    return OpenCLEvent( event );
  }
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLKernel::LaunchKernel( const OpenCLSize & global_work_size,
  const OpenCLSize & local_work_size,
  const OpenCLSize & global_work_offset )
{
  this->SetGlobalWorkSize( global_work_size );
  this->SetLocalWorkSize( local_work_size );
  this->SetGlobalWorkOffset( global_work_offset );
  return this->LaunchKernel();
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLKernel::LaunchKernel( const OpenCLEventList & event_list )
{
  ITK_OPENCL_D( const OpenCLKernel );

  const cl_uint work_dim = d->global_work_size.GetDimension();
  const bool    gwoNull  = d->global_work_offset.IsNull();
  const bool    lwsNull  = d->local_work_size.IsNull();

  cl_event event;
  cl_int   error;

  if( gwoNull && lwsNull )
  {
    error = clEnqueueNDRangeKernel( d->context->GetActiveQueue(), this->m_KernelId,
      work_dim, NULL, d->global_work_size.GetSizes(), NULL,
      event_list.GetSize(), event_list.GetEventData(), &event );
  }
  else if( gwoNull && !lwsNull )
  {
    error = clEnqueueNDRangeKernel( d->context->GetActiveQueue(), this->m_KernelId,
      work_dim, NULL, d->global_work_size.GetSizes(),
      ( d->local_work_size.GetWidth() ? d->local_work_size.GetSizes() : 0 ),
      event_list.GetSize(), event_list.GetEventData(), &event );
  }
  else if( !gwoNull && lwsNull )
  {
    error = clEnqueueNDRangeKernel( d->context->GetActiveQueue(), this->m_KernelId,
      work_dim, d->global_work_offset.GetSizes(), d->global_work_size.GetSizes(),
      NULL,
      event_list.GetSize(), event_list.GetEventData(), &event );
  }
  else
  {
    error = clEnqueueNDRangeKernel( d->context->GetActiveQueue(), this->m_KernelId,
      work_dim, d->global_work_offset.GetSizes(), d->global_work_size.GetSizes(),
      ( d->local_work_size.GetWidth() ? d->local_work_size.GetSizes() : 0 ),
      event_list.GetSize(), event_list.GetEventData(), &event );
  }

  if( error != CL_SUCCESS )
  {
    //const std::size_t num = GetNumberOfArguments();
    itkOpenCLErrorMacroGeneric( << "Launch kernel '" << this->GetName() << "' failed." );
    d->context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
    return OpenCLEvent();
  }
  else
  {
#ifdef OPENCL_PROFILING
    const std::string profileStr = "clEnqueueNDRangeKernel: " + this->GetName();
    d->context->OpenCLProfile( event, profileStr );
#endif
    return OpenCLEvent( event );
  }
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLKernel::LaunchKernel( const OpenCLEventList & event_list,
  const OpenCLSize & global_work_size,
  const OpenCLSize & local_work_size,
  const OpenCLSize & global_work_offset )
{
  this->SetGlobalWorkSize( global_work_size );
  this->SetLocalWorkSize( local_work_size );
  this->SetGlobalWorkOffset( global_work_offset );
  return this->LaunchKernel( event_list );
}


//------------------------------------------------------------------------------
bool
OpenCLKernel::LaunchTask( const OpenCLEventList & event_list )
{
  if( event_list.IsEmpty() )
  {
    return true;
  }

  ITK_OPENCL_D( const OpenCLKernel );
  cl_event     event;
  const cl_int error = clEnqueueTask( d->context->GetActiveQueue(), this->m_KernelId,
    0, 0, &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( error == CL_SUCCESS )
  {
    clWaitForEvents( 1, &event );
    clReleaseEvent( event );
    return true;
  }
  else
  {
    return false;
  }
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLKernel::LaunchTaskAsync( const OpenCLEventList & event_list )
{
  if( event_list.IsEmpty() )
  {
    return OpenCLEvent();
  }

  ITK_OPENCL_D( const OpenCLKernel );
  cl_event     event;
  const cl_int error = clEnqueueTask( d->context->GetActiveQueue(), this->m_KernelId,
    0, 0, &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( error == CL_SUCCESS )
  {
    return OpenCLEvent( event );
  }
  else
  {
    return OpenCLEvent();
  }
}


//------------------------------------------------------------------------------
//! Operator ==
bool
operator==( const OpenCLKernel & lhs, const OpenCLKernel & rhs )
{
  if( &rhs == &lhs )
  {
    return true;
  }
  return lhs.GetKernelId() == rhs.GetKernelId();
}


//------------------------------------------------------------------------------
//! Operator !=
bool
operator!=( const OpenCLKernel & lhs, const OpenCLKernel & rhs )
{
  return !( lhs == rhs );
}


} // namespace itk
