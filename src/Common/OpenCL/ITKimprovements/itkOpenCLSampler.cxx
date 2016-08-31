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
#include "itkOpenCLSampler.h"
#include "itkOpenCLContext.h"

namespace itk
{
OpenCLSampler::OpenCLSampler( const OpenCLSampler & other ) :
  m_Context( other.m_Context ), m_Id( other.m_Id )
{
  if( !this->IsNull() )
  {
    clRetainSampler( this->m_Id );
  }
}


//------------------------------------------------------------------------------
OpenCLSampler::~OpenCLSampler()
{
  if( !this->IsNull() )
  {
    clReleaseSampler( this->m_Id );
  }
}


//------------------------------------------------------------------------------
OpenCLSampler &
OpenCLSampler::operator=( const OpenCLSampler & other )
{
  this->m_Context = other.m_Context;
  if( other.m_Id )
  {
    clRetainSampler( other.m_Id );
  }
  if( this->m_Id )
  {
    clReleaseSampler( this->m_Id );
  }
  this->m_Id = other.m_Id;
  return *this;
}


//------------------------------------------------------------------------------
bool
OpenCLSampler::GetNormalizedCoordinates() const
{
  if( !this->IsNull() )
  {
    cl_bool normalized = CL_FALSE;
    clGetSamplerInfo( this->m_Id, CL_SAMPLER_NORMALIZED_COORDS,
      sizeof( normalized ), &normalized, 0 );
    return normalized != CL_FALSE;
  }
  else
  {
    return false;
  }
}


//------------------------------------------------------------------------------
OpenCLSampler::AddressingMode
OpenCLSampler::GetAddressingMode() const
{
  if( !this->IsNull() )
  {
    cl_addressing_mode addressing;
    clGetSamplerInfo( this->m_Id, CL_SAMPLER_ADDRESSING_MODE,
      sizeof( addressing ), &addressing, 0 );
    return OpenCLSampler::AddressingMode( addressing );
  }
  else
  {
    return ClampToEdge;
  }
}


//------------------------------------------------------------------------------
OpenCLSampler::FilterMode
OpenCLSampler::GetFilterMode() const
{
  if( !this->IsNull() )
  {
    cl_filter_mode filter;
    clGetSamplerInfo( this->m_Id, CL_SAMPLER_FILTER_MODE,
      sizeof( filter ), &filter, 0 );
    return OpenCLSampler::FilterMode( filter );
  }
  else
  {
    return Linear;
  }
}


//------------------------------------------------------------------------------
//! Operator ==
bool
operator==( const OpenCLSampler & lhs, const OpenCLSampler & rhs )
{
  if( &rhs == &lhs )
  {
    return true;
  }
  return lhs.GetSamplerId() == rhs.GetSamplerId();
}


//------------------------------------------------------------------------------
//! Operator !=
bool
operator!=( const OpenCLSampler & lhs, const OpenCLSampler & rhs )
{
  return !( lhs == rhs );
}


} // end namespace itk
