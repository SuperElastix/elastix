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
#include "itkOpenCLEventList.h"
#include "itkOpenCLMacro.h"
#include "itkOpenCLContext.h"

namespace itk
{
OpenCLEventList::OpenCLEventList( const OpenCLEvent & event )
{
  cl_event id = event.GetEventId();

  if( id )
  {
    clRetainEvent( id );
    this->m_Events.push_back( id );
  }
}


//------------------------------------------------------------------------------
OpenCLEventList::OpenCLEventList( const OpenCLEventList & other ) :
  m_Events( other.m_Events )
{
  for( std::size_t index = 0; index < this->m_Events.size(); ++index )
  {
    clRetainEvent( this->m_Events[ index ] );
  }
}


//------------------------------------------------------------------------------
OpenCLEventList::~OpenCLEventList()
{
  for( std::size_t index = 0; index < this->m_Events.size(); ++index )
  {
    clReleaseEvent( this->m_Events[ index ] );
  }
}


//------------------------------------------------------------------------------
OpenCLEventList &
OpenCLEventList::operator=( const OpenCLEventList & other )
{
  if( this != &other )
  {
    for( std::size_t index = 0; index < this->m_Events.size(); ++index )
    {
      clReleaseEvent( this->m_Events[ index ] );
    }
    this->m_Events = other.m_Events;
    for( std::size_t index = 0; index < this->m_Events.size(); ++index )
    {
      clRetainEvent( this->m_Events[ index ] );
    }
  }
  return *this;
}


//------------------------------------------------------------------------------
void
OpenCLEventList::Append( const OpenCLEvent & event )
{
  cl_event id = event.GetEventId();

  if( id )
  {
    clRetainEvent( id );
    this->m_Events.push_back( id );
  }
}


//------------------------------------------------------------------------------
void
OpenCLEventList::Append( const OpenCLEventList & other )
{
  for( std::size_t index = 0; index < other.m_Events.size(); ++index )
  {
    cl_event id = other.m_Events[ index ];
    clRetainEvent( id );
    this->m_Events.push_back( id );
  }
}


//------------------------------------------------------------------------------
void
OpenCLEventList::Remove( const OpenCLEvent & event )
{
  OpenCLEventListArrayType::iterator it;

  for( it = this->m_Events.begin(); it < this->m_Events.end(); it++ )
  {
    if( *it == event.GetEventId() )
    {
      clReleaseEvent( *it );
      this->m_Events.erase( it );
    }
  }
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLEventList::Get( const std::size_t index ) const
{
  if( index < this->m_Events.size() )
  {
    cl_event id = this->m_Events[ index ];
    clRetainEvent( id );
    return OpenCLEvent( id );
  }
  else
  {
    return OpenCLEvent();
  }
}


//------------------------------------------------------------------------------
bool
OpenCLEventList::Contains( const OpenCLEvent & event ) const
{
  OpenCLEventListArrayType::const_iterator it;

  for( it = this->m_Events.begin(); it < this->m_Events.end(); it++ )
  {
    if( *it == event.GetEventId() )
    {
      return true;
    }
  }
  return false;
}


//------------------------------------------------------------------------------
const cl_event *
OpenCLEventList::GetEventData() const
{
  return this->m_Events.empty() ? 0 : &this->m_Events[ 0 ];
}


//------------------------------------------------------------------------------
const OpenCLEventList::OpenCLEventListArrayType &
OpenCLEventList::GetEventArray() const
{
  return this->m_Events;
}


//------------------------------------------------------------------------------
OpenCLEventList &
OpenCLEventList::operator+=( const OpenCLEvent & event )
{
  this->Append( event );
  return *this;
}


//------------------------------------------------------------------------------
OpenCLEventList &
OpenCLEventList::operator+=( const OpenCLEventList & other )
{
  this->Append( other );
  return *this;
}


//------------------------------------------------------------------------------
OpenCLEventList &
OpenCLEventList::operator<<( const OpenCLEvent & event )
{
  this->Append( event );
  return *this;
}


//------------------------------------------------------------------------------
OpenCLEventList &
OpenCLEventList::operator<<( const OpenCLEventList & other )
{
  this->Append( other );
  return *this;
}


//------------------------------------------------------------------------------
cl_int
OpenCLEventList::WaitForFinished()
{
  if( this->m_Events.empty() )
  {
    return 0;
  }

  const cl_int error = clWaitForEvents( this->GetSize(), this->GetEventData() );
  if( error != CL_SUCCESS )
  {
    itkOpenCLErrorMacroGeneric( << "OpenCLEventList::WaitForFinished:"
                                << OpenCLContext::GetErrorName( error ) );
  }
  return error;
}


//------------------------------------------------------------------------------
//! Operator ==
bool
operator==( const OpenCLEventList & lhs, const OpenCLEventList & rhs )
{
  if( &rhs == &lhs )
  {
    return true;
  }

  const OpenCLEventList::OpenCLEventListArrayType & eventsArrayLHS = lhs.GetEventArray();
  const OpenCLEventList::OpenCLEventListArrayType & eventsArrayRHS = rhs.GetEventArray();

  if( eventsArrayLHS.size() != eventsArrayRHS.size() )
  {
    return false;
  }

  OpenCLEventList::OpenCLEventListArrayType::const_iterator ilhs;
  OpenCLEventList::OpenCLEventListArrayType::const_iterator irhs;
  for( ilhs = eventsArrayLHS.begin(), irhs = eventsArrayRHS.begin();
    ilhs < eventsArrayLHS.end() && irhs < eventsArrayRHS.end();
    ++ilhs, ++irhs )
  {
    if( *ilhs != *irhs )
    {
      return false;
    }
  }

  return true;
}


//------------------------------------------------------------------------------
//! Operator !=
bool
operator!=( const OpenCLEventList & lhs, const OpenCLEventList & rhs )
{
  return !( lhs == rhs );
}


} // namespace itk
