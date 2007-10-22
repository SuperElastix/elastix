#include "elxProgressCommand.h"
#include "xoutmain.h"
#include "vnl/vnl_math.h"


namespace elastix
{
  using namespace itk;


  /**
	 * ******************* Constructor ***********************
	 */

  ProgressCommand::ProgressCommand()
  {
    this->m_StartString = "Progress: ";
    this->m_EndString = "%";

    /** Check if the output of the stream is a console. */
    this->m_StreamOutputIsConsole = false;
    std::string streamOutput = "cout";
    int currentPos = xl::xout["coutonly"].GetCOutputs().find( streamOutput )->second->tellp();
    if ( currentPos == -1 )
    {
      this->m_StreamOutputIsConsole = true;
    }

  } // end Constructor()


  /**
	 * ******************* ConnectObserver ***********************
	 */

  void ProgressCommand
    ::ConnectObserver( ProcessObject * filter )
  {
    if ( this->m_StreamOutputIsConsole )
    {
      filter->AddObserver( itk::ProgressEvent(), this );
    }
  } // end ConnectObserver()


  /**
	 * ******************* Execute ***********************
	 */

  void ProgressCommand
    ::Execute( Object *caller, const EventObject &event )
  {
    ProcessObject *po = dynamic_cast<ProcessObject *>( caller );
    if ( !po ) return;

    if ( typeid( event ) == typeid( ProgressEvent ) )
    {
      this->PrintProgress( po->GetProgress() );
    }

  } // end Execute()


  /**
	 * ******************* Execute ***********************
	 */
  
  void ProgressCommand
    ::Execute( const Object *caller, const EventObject &event )
  {
    const ProcessObject *po = dynamic_cast<const ProcessObject *>( caller );
    if ( !po ) return;
  
    if ( typeid( event ) == typeid( ProgressEvent ) )
    {
      this->PrintProgress( po->GetProgress() );
    }

  } // end Execute()


  /**
	 * ******************* PrintProgress ***********************
	 */

  void ProgressCommand
    ::PrintProgress( const float & progress ) const
  {
    /** Print the progress to the screen. */
    int progressInt = vnl_math_rnd( 100 * progress );
    xl::xout["coutonly"]
      << "\r"
      << this->m_StartString
      << progressInt
      << this->m_EndString;

    /** If the process is completed, print an end-of-line. */
    if ( progress > 0.99999 )
    {
      xl::xout["coutonly"] << std::endl;
    }

  } // end PrintProgress()


} // end namespace itk

