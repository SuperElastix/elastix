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
#include "itkOpenCLLogger.h"

#include "itkLogger.h"
#include <sstream>

namespace itk
{
// static variable initialization
OpenCLLogger::Pointer OpenCLLogger::m_Instance = 0;

//------------------------------------------------------------------------------
// Return the single instance of the OpenCLLogger
OpenCLLogger::Pointer
OpenCLLogger::GetInstance()
{
  if( !OpenCLLogger::m_Instance )
  {
    // Try the factory first
    OpenCLLogger::m_Instance = ObjectFactory< Self >::Create();
    // if the factory did not provide one, then create it here
    if( !OpenCLLogger::m_Instance )
    {
      // For the windows OS, use a special output window
      OpenCLLogger::m_Instance = new OpenCLLogger;
      // Remove extra reference from construction.
      OpenCLLogger::m_Instance->UnRegister();
    }
  }
  // Return the instance
  return OpenCLLogger::m_Instance;
}


//------------------------------------------------------------------------------
// This just calls GetInstance
OpenCLLogger::Pointer
OpenCLLogger::New()
{
  return GetInstance();
}


//------------------------------------------------------------------------------
OpenCLLogger::OpenCLLogger()
{
  this->m_FileName   = "_opencl.log";
  this->m_FileStream = NULL;
  this->m_Created    = false;
}


//------------------------------------------------------------------------------
OpenCLLogger::~OpenCLLogger()
{
  // Would prefer to close the m_FileStream, but StdStreamLogOutput
  // still performs extra flush in the destructor and setting it with null
  // not possible due to SetStream() implementation.

  //if( this->m_FileStream != NULL )
  //{
  //  delete this->m_FileStream;
  //  this->m_FileStream = NULL;
  //}
}


//------------------------------------------------------------------------------
void
OpenCLLogger::Initialize()
{
  // Construct log filename
  const std::string forward_slash( "/" );
  std::string       logFileName = this->m_OutputDirectory;
  const std::size_t found       = logFileName.find_last_not_of( forward_slash );
  if( found == std::string::npos )
  {
    logFileName.append( "/" );
    logFileName.append( this->m_FileName );
  }
  else
  {
    logFileName.append( this->m_FileName );
  }

  // Create file stream
  this->m_FileStream = new std::ofstream( logFileName.c_str(), std::ios::out );
  if( this->m_FileStream->fail() )
  {
    itkExceptionMacro( << "Unable to open file: " << logFileName );
    delete this->m_FileStream;
    this->m_FileStream = NULL;
    this->m_Created    = false;
    return;
  }

  // Create an ITK Logger
  LoggerBase::TimeStampFormatType timeStampFormat = LoggerBase::HUMANREADABLE;
  this->SetTimeStampFormat( timeStampFormat );
  const std::string humanReadableFormat = "%b %d %Y %H:%M:%S";
  this->SetHumanReadableFormat( humanReadableFormat );

  // Setting the logger
  this->SetName( "OpenCLLogger" );
  this->SetPriorityLevel( LoggerBase::INFO );
  this->SetLevelForFlushing( LoggerBase::CRITICAL );

  // Create StdStreamLogOutput
  this->m_Stream = StdStreamLogOutput::New();
  this->m_Stream->SetStream( *this->m_FileStream );

  // Add to logger
  this->AddLogOutput( this->m_Stream );

  this->m_Created = true;
}


//------------------------------------------------------------------------------
void
OpenCLLogger::SetLogFileNamePrefix( const std::string & prefix )
{
  this->m_FileName.insert( 0, prefix );
}


//------------------------------------------------------------------------------
bool
OpenCLLogger::IsCreated() const
{
  return this->m_Created;
}


//------------------------------------------------------------------------------
void
OpenCLLogger::Write( PriorityLevelType level, std::string const & content )
{
  if( this->m_Stream.IsNull() )
  {
    this->Initialize();
  }

  if( !this->IsCreated() )
  {
    return;
  }

  std::ostringstream message;
  message << "OpenCL compile error: " << std::endl << content;
  Superclass::Write( level, message.str().c_str() );
}


}
