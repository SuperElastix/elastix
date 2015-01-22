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
#include "itkOpenCLUserEvent.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLMacro.h"

namespace itk
{
OpenCLUserEvent::OpenCLUserEvent( cl_event id ) :
  OpenCLEvent( id )
{
  this->ReleaseIfNotUserEvent();
}


//------------------------------------------------------------------------------
OpenCLUserEvent::OpenCLUserEvent( const OpenCLEvent & other ) :
  OpenCLEvent( other )
{
  this->ReleaseIfNotUserEvent();
}


//------------------------------------------------------------------------------
OpenCLUserEvent &
OpenCLUserEvent::operator=( const OpenCLEvent & other )
{
  if( this->m_Id != other.m_Id )
  {
    if( this->m_Id )
    {
      clReleaseEvent( this->m_Id );
    }
    this->m_Id = other.m_Id;
    if( this->m_Id )
    {
      clRetainEvent( this->m_Id );
    }
    this->ReleaseIfNotUserEvent();
  }
  return *this;
}


//------------------------------------------------------------------------------
void
OpenCLUserEvent::SetComplete()
{
  this->SetStatus( CL_COMPLETE );
}


//------------------------------------------------------------------------------
void
OpenCLUserEvent::SetStatus( const cl_int status )
{
  if( this->m_Id )
  {
    cl_int error = clSetUserEventStatus( this->m_Id, status );
    if( error != CL_SUCCESS )
    {
      itkOpenCLWarningMacroGeneric( << "OpenCLUserEvent::SetStatus:"
                                    << OpenCLContext::GetErrorName( error ) );
    }
  }
}


//------------------------------------------------------------------------------
void
OpenCLUserEvent::ReleaseIfNotUserEvent()
{
  if( this->m_Id && this->GetCommandType() != CL_COMMAND_USER )
  {
    clReleaseEvent( this->m_Id );
    this->m_Id = 0;
  }
}


} // namespace itk
