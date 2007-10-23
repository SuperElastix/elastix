#ifndef __elxProgressCommand_h
#define __elxProgressCommand_h

#include "itkProcessObject.h"
#include "itkCommand.h"

namespace elastix
{
	using namespace itk;
	
	/**
	 * \class ProgressCommand
	 * \brief A specialized Command object for updating the progress of a
   *  filter.
	 *
   * Whenever a filter, such as the itk::ResampleImageFilter, supports
   * a ProgressReporter, this class can be employed. This class makes
   * sure that the progress of a filter is printed to screen. It works
   * as follows:
   *
   *   ProgressCommandType::Pointer command = ProgressCommandType::New();
   *   command->ConnectObserver( filterpointer );
   *   command->SetStartString( "  Progress: " );
   *   command->SetEndString( "%" );
   *
   * So, first an instantiation of this class is created, then it is
   * connected to a filter, and some options are set. Whenever the filter
   * throws a ProgressEvent(), this class asks for the progress and prints
   * the percentage of progress.
   *
	 * //\ingroup Resamplers
	 */

class ProgressCommand : public Command
{
public:

  /** Smart pointer declaration methods */
  typedef ProgressCommand               Self;
  typedef Command                       Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Standard ITK stuff. */
  itkTypeMacro( ProgressCommand, Command );
  itkNewMacro( Self );

  /** Typedef's. */
  typedef ProcessObject                 ProcessObjectType;
  typedef ProcessObjectType::Pointer    ProcessObjectPointer;

  /** Define when to print the progress. */
  virtual void SetUpdateFrequency(
    const unsigned long numberOfVoxels,
    const unsigned long numberOfUpdates );

  /** Connect an observer to a process object. */
  virtual void ConnectObserver( ProcessObject * filter );

  /** Disconnect an observer to a process object. */
  virtual void DisconnectObserver( ProcessObject * filter );

  /** Standard Command virtual methods. */
  virtual void Execute( Object *caller, const EventObject &event );
  virtual void Execute( const Object *caller, const EventObject &event );

  /** Print the progress to screen. */
  virtual void PrintProgress( const float & progress ) const;

  /** Print the progress to screen. */
  virtual void PrintProgress( const unsigned long & currentVoxelNumber ) const;

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
  bool                  m_StreamOutputIsConsole;
  unsigned long         m_Tag;
  bool                  m_TagIsSet;
  ProcessObjectPointer  m_ObservedProcessObject;

  /** Member variables that define the update frequency. */
  unsigned long m_NumberOfVoxels;
  unsigned long m_NumberOfUpdates;
  
}; // end class ProgressCommand

} // end namespace elastix

#endif // end #ifndef __elxProgressCommand_h
