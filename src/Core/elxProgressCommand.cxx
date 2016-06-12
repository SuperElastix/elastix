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

#include "elxProgressCommand.h"
#include "xoutmain.h"
#include "itkMath.h" // itk::Math::Round

namespace elastix
{
/**
 * ******************* Constructor ***********************
 */

ProgressCommand::ProgressCommand()
{
  this->m_StartString           = "Progress: ";
  this->m_EndString             = "%";
  this->m_Tag                   = 0;
  this->m_TagIsSet              = false;
  this->m_ObservedProcessObject = 0;
  this->m_NumberOfVoxels        = 0;
  this->m_NumberOfUpdates       = 0;

  /** Check if the output of the stream is a console. */
  this->m_StreamOutputIsConsole = false;
  std::string streamOutput = "cout";
  int         currentPos   = xl::xout[ "coutonly" ].GetCOutputs().find( streamOutput )->second->tellp();
  if( currentPos == -1 )
  {
    this->m_StreamOutputIsConsole = true;
  }

}   // end Constructor()


/**
 * ******************* Destructor ***********************
 */

ProgressCommand::~ProgressCommand()
{
  this->DisconnectObserver( this->m_ObservedProcessObject );

}   // end Destructor()


/**
 * ******************* SetUpdateFrequency ***********************
 */

void
ProgressCommand
::SetUpdateFrequency(
  const unsigned long numberOfVoxels,
  const unsigned long numberOfUpdates )
{
  /** Set the member variables. */
  this->m_NumberOfVoxels  = numberOfVoxels;
  this->m_NumberOfUpdates = numberOfUpdates;

  /** Make sure we have at least one pixel. */
  if( this->m_NumberOfVoxels < 1 )
  {
    this->m_NumberOfVoxels = 1;
  }

  /** We cannot update more times than there are pixels. */
  if( this->m_NumberOfUpdates > this->m_NumberOfVoxels )
  {
    this->m_NumberOfUpdates = this->m_NumberOfVoxels;
  }

  /** Make sure we update at least once. */
  if( this->m_NumberOfUpdates < 1 )
  {
    this->m_NumberOfUpdates = 1;
  }

}   // end SetUpdateFrequency()


/**
 * ******************* ConnectObserver ***********************
 */

void
ProgressCommand
::ConnectObserver( itk::ProcessObject * filter )
{
  /** Disconnect from old observed filters. */
  this->DisconnectObserver( this->m_ObservedProcessObject );

  /** Connect to the new filter. */
  if( this->m_StreamOutputIsConsole )
  {
    this->m_Tag                   = filter->AddObserver( itk::ProgressEvent(), this );
    this->m_TagIsSet              = true;
    this->m_ObservedProcessObject = filter;
  }

}   // end ConnectObserver()


/**
 * ******************* DisconnectObserver ***********************
 */

void
ProgressCommand
::DisconnectObserver( itk::ProcessObject * filter )
{
  if( this->m_StreamOutputIsConsole )
  {
    if( this->m_TagIsSet )
    {
      filter->RemoveObserver( this->m_Tag );
      this->m_TagIsSet              = false;
      this->m_ObservedProcessObject = 0;
    }
  }

}   // end DisconnectObserver()


/**
 * ******************* Execute ***********************
 */

void
ProgressCommand
::Execute( itk::Object * caller, const itk::EventObject & event )
{
  itk::ProcessObject * po = dynamic_cast< itk::ProcessObject * >( caller );
  if( !po ) { return; }

  if( typeid( event ) == typeid( itk::ProgressEvent ) )
  {
    this->PrintProgress( po->GetProgress() );
  }

}   // end Execute()


/**
 * ******************* Execute ***********************
 */

void
ProgressCommand
::Execute( const itk::Object * caller, const itk::EventObject & event )
{
  const itk::ProcessObject * po = dynamic_cast< const itk::ProcessObject * >( caller );
  if( !po ) { return; }

  if( typeid( event ) == typeid( itk::ProgressEvent ) )
  {
    this->PrintProgress( po->GetProgress() );
  }

}   // end Execute()


/**
 * ******************* PrintProgress ***********************
 */

void
ProgressCommand
::PrintProgress( const float & progress ) const
{
  /** Print the progress to the screen. */
  const int progressInt = itk::Math::Round< float >( 100 * progress );
  xl::xout[ "coutonly" ]
    << "\r"
    << this->m_StartString
    << progressInt
    << this->m_EndString;
  xl::xout[ "coutonly" ] << std::flush;

  /** If the process is completed, print an end-of-line. *
  if ( progress > 0.99999 )
  {
    xl::xout["coutonly"] << std::endl;
  }*/

}   // end PrintProgress()


/**
 * ******************* PrintProgress ***********************
 */

void
ProgressCommand
::UpdateAndPrintProgress( const unsigned long & currentVoxelNumber ) const
{
  if( this->m_StreamOutputIsConsole )
  {
    const unsigned long frac
      = static_cast< unsigned long >( this->m_NumberOfVoxels / this->m_NumberOfUpdates );
    if( currentVoxelNumber % frac == 0 )
    {
      this->PrintProgress(
        static_cast< float >( currentVoxelNumber )
        / static_cast< float >( this->m_NumberOfVoxels ) );
    }
  }

}   // end PrintProgress()


} // end namespace itk
