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
#include "itkOpenCLCommandQueue.h"
#include "itkOpenCLContext.h"

namespace itk
{
OpenCLCommandQueue::OpenCLCommandQueue( const OpenCLCommandQueue & other ) :
  m_Context( other.m_Context ), m_Id( other.m_Id )
{
  if( !this->IsNull() )
  {
    clRetainCommandQueue( this->m_Id );
  }
}


//------------------------------------------------------------------------------
OpenCLCommandQueue::~OpenCLCommandQueue()
{
  if( !this->IsNull() )
  {
    clReleaseCommandQueue( this->m_Id );
  }
}


//------------------------------------------------------------------------------
OpenCLCommandQueue &
OpenCLCommandQueue::operator=( const OpenCLCommandQueue & other )
{
  this->m_Context = other.m_Context;
  if( other.m_Id )
  {
    clRetainCommandQueue( other.m_Id );
  }
  if( this->m_Id )
  {
    clReleaseCommandQueue( this->m_Id );
  }
  this->m_Id = other.m_Id;
  return *this;
}


//------------------------------------------------------------------------------
bool
OpenCLCommandQueue::IsOutOfOrder() const
{
  if( this->IsNull() )
  {
    return false;
  }
  cl_command_queue_properties props = 0;
  if( clGetCommandQueueInfo( this->m_Id, CL_QUEUE_PROPERTIES,
    sizeof( props ), &props, 0 ) != CL_SUCCESS )
  {
    return false;
  }
  return ( props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ) != 0;
}


//------------------------------------------------------------------------------
bool
OpenCLCommandQueue::IsProfilingEnabled() const
{
  if( this->IsNull() )
  {
    return false;
  }
  cl_command_queue_properties props = 0;
  if( clGetCommandQueueInfo( this->m_Id, CL_QUEUE_PROPERTIES,
    sizeof( props ), &props, 0 ) != CL_SUCCESS )
  {
    return false;
  }
  return ( props & CL_QUEUE_PROFILING_ENABLE ) != 0;
}


//------------------------------------------------------------------------------
//! Operator ==
bool
operator==( const OpenCLCommandQueue & lhs, const OpenCLCommandQueue & rhs )
{
  if( &rhs == &lhs )
  {
    return true;
  }
  return lhs.GetQueueId() == rhs.GetQueueId();
}


//------------------------------------------------------------------------------
//! Operator !=
bool
operator!=( const OpenCLCommandQueue & lhs, const OpenCLCommandQueue & rhs )
{
  return !( lhs == rhs );
}


} // namespace itk
