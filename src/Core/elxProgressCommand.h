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

  /** Connect an observer to a process object. */
  virtual void ConnectObserver( ProcessObject * filter );

  /** Standard Command virtual methods. */
  virtual void Execute( Object *caller, const EventObject &event );
  virtual void Execute( const Object *caller, const EventObject &event );

  /** Print the progress to screen. */
  virtual void PrintProgress( const float & progress ) const;

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
  virtual ~ProgressCommand() {}
  
private:

  bool m_StreamOutputIsConsole;
  std::string m_StartString;
  std::string m_EndString;
  
}; // end class ProgressCommand

} // end namespace elastix

#endif // end #ifndef __elxProgressCommand_h
