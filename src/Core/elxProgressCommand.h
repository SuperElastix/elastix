/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxProgressCommand_h
#define __elxProgressCommand_h

#include "itkProcessObject.h"
#include "itkCommand.h"

namespace elastix
{

/**
 * \class ProgressCommand
 * \brief A specialized Command object for updating the progress of a
 *  filter.
 *
 * There are 3 ways to use this class.
 *
 * \li Whenever a filter, such as the itk::ResampleImageFilter, supports
 * a ProgressReporter, this class can be employed. This class makes
 * sure that the progress of a filter is printed to screen. It works
 * as follows:
 *
 * \code
 *   ProgressCommandType::Pointer command = ProgressCommandType::New();
 *   command->ConnectObserver( filterpointer );
 *   command->SetStartString( "  Progress: " );
 *   command->SetEndString( "%" );
 *   filterpointer->Update(); // run the filter, progress messages are printed now
 *   command->DisconnectObserver( filterPointer );
 * \endcode
 *
 * So, first an instantiation of this class is created, then it is
 * connected to a filter, and some options are set. Whenever the filter
 * throws a ProgressEvent(), this class asks for the progress and prints
 * the percentage of progress.
 *
 * \li In manually written loops, a call to UpdateAndPrintProgress() can be included.
 * Before the loop, the user should set the total number of loops, and the frequency
 * that the progress message should be printed with. For example
 *
 * \code
 *   ProgressCommandType::Pointer command = ProgressCommandType::New();
 *   command->SetUpdateFrequency( maxnrofvoxels, 100 );
 *   command->SetStartString( "  Progress: " );
 *   command->SetEndString( "%" );
 *   elxout << "Looping over voxels... " << std::endl;
 *   for ( unsigned int i =0; i < maxnrofvoxels; ++i )
 *   {
 *     command->UpdateAndPrintProgress( i );
 *   }
 *   command->PrintProgress(1.0); // make sure the 100% is reached
 * \endcode
 *
 * \li The last possibility is to directly use the PrintProgress function:
 *
 * \code
 *   ProgressCommandType::Pointer command = ProgressCommandType::New();
 *   command->SetStartString( "  Progress: " );
 *   command->SetEndString( "%" );
 *   elxout << "Reading, casting, writing..."
 *   command->PrintProgress( 0.0 );
 *   reader->Update();
 *   command->PrintProgress( 0.33 );
 *   caster->Update();
 *   command->PrintProgress( 0.67 );
 *   writer->Update();
 *   command->PrintProgress( 1.0 );
 *   // example assumes reader, caster and writer have been configured before
 * \endcode
 *
 */

class ProgressCommand : public itk::Command
{
public:

  /** Smart pointer declaration methods. */
  typedef ProgressCommand                 Self;
  typedef itk::Command                    Superclass;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Standard ITK stuff. */
  itkTypeMacro( ProgressCommand, Command );
  itkNewMacro( Self );

  /** Typedef's. */
  typedef itk::ProcessObject         ProcessObjectType;
  typedef ProcessObjectType::Pointer ProcessObjectPointer;

  /** Define when to print the progress. */
  virtual void SetUpdateFrequency(
    const unsigned long numberOfVoxels,
    const unsigned long numberOfUpdates );

  /** Connect an observer to a process object. */
  virtual void ConnectObserver( itk::ProcessObject * filter );

  /** Disconnect an observer to a process object. */
  virtual void DisconnectObserver( itk::ProcessObject * filter );

  /** Standard Command virtual methods. */
  virtual void Execute( itk::Object * caller, const itk::EventObject & event );

  virtual void Execute( const itk::Object * caller, const itk::EventObject & event );

  /** Print the progress to screen. A float value between 0.0 and 1.0
   * is expected as input.
   */
  virtual void PrintProgress( const float & progress ) const;

  /** Update and possibly print the progress to screen.
   * The progress information on screen is refreshed according to the
   * UpdateFrequency, which is assumed being specified beforehand using the
   * SetUpdateFrequency function.
   */
  virtual void UpdateAndPrintProgress( const unsigned long & currentVoxelNumber ) const;

  /** Set and get the string starting each progress report. */
  itkSetStringMacro( StartString );
  itkGetStringMacro( StartString );

  /** Set and get the string ending each progress report. */
  itkSetStringMacro( EndString );
  itkGetStringMacro( EndString );

  /** Get a boolean indicating if the output is a console. */
  itkGetConstReferenceMacro( StreamOutputIsConsole, bool );

protected:

  /** The constructor. */
  ProgressCommand();

  /** The destructor. */
  virtual ~ProgressCommand();

private:

  /** Member variables to define a start and end string for printing. */
  std::string m_StartString;
  std::string m_EndString;

  /** Member variables to keep track of what is set. */
  bool                 m_StreamOutputIsConsole;
  unsigned long        m_Tag;
  bool                 m_TagIsSet;
  ProcessObjectPointer m_ObservedProcessObject;

  /** Member variables that define the update frequency. */
  unsigned long m_NumberOfVoxels;
  unsigned long m_NumberOfUpdates;

};

} // end namespace elastix

#endif // end #ifndef __elxProgressCommand_h
