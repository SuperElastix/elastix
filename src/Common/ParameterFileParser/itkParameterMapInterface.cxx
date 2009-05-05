#ifndef __itkParameterMapInterface_cxx
#define __itkParameterMapInterface_cxx

#include "itkParameterMapInterface.h"


namespace itk
{

/**
 * **************** Constructor ***************
 */
  
ParameterMapInterface
::ParameterMapInterface()
{
  this->m_ParameterMap.clear();
  this->m_PrintErrorMessages = true;
    
} // end Constructor()


/**
 * **************** Destructor ***************
 */
  
ParameterMapInterface
::~ParameterMapInterface()
{
    // empty
} // end Destructor()


/**
 * **************** SetParameterMap ***************
 */

void
ParameterMapInterface
::SetParameterMap( const ParameterMapType & parMap )
{
  if ( !parMap.empty() )
  {
    this->m_ParameterMap = parMap;
  }

} // end SetParameterMap()


/**
 * **************** CountNumberOfParameterEntries ***************
 */

unsigned int
ParameterMapInterface
::CountNumberOfParameterEntries(
  const std::string & parameterName ) const
{
  if ( this->m_ParameterMap.count( parameterName ) )
  {
    return this->m_ParameterMap.find( parameterName )->second.size();
  }
  return 0;
  
} // end CountNumberOfParameterEntries()


/**
 * **************** ReadParameter ***************
 */

bool
ParameterMapInterface
::ReadParameter( bool & parameterValue,
  const std::string & parameterName,
  const unsigned int entry_nr,
  const bool printThisErrorMessage,
  std::string & errorMessage ) const
{
  /** Translate the default boolean to string. */
  std::string parameterValueString;
  if ( parameterValue )
  {
    parameterValueString = "true";
  }
  else
  {
    parameterValueString = "false";
  }

  /** Read the boolean as a string. */
  bool dummy = this->ReadParameter( parameterValueString, parameterName,
    entry_nr, printThisErrorMessage, errorMessage );

  /** Translate the read-in string to boolean. */
  parameterValue = false;
  if ( parameterValueString == "true" )
  {
    parameterValue = true;
  }
  else if ( parameterValueString == "false" )
  {
    parameterValue = false;
  }
  else
  {
    /** Trying to read a string other than "true" or "false" as a boolean. */
    std::stringstream ss;
    ss << "ERROR: Entry number " << entry_nr
      << " for the parameter \"" << parameterName
      << "\" should be a boolean, i.e. either \"true\" or \"false\""
      << ", but it reads \"" << parameterValueString << "\".";

    itkExceptionMacro( << ss.str() );
  }

  return dummy;

} // end ReadParameter()


/**
 * **************** StringCast ***************
 */

bool
ParameterMapInterface
::StringCast( const std::string & parameterValue, std::string & casted ) const
{
  casted = parameterValue;
  return true;
} // end StringCast()


} // end namespace itk

#endif // end __itkParameterMapInterface_cxx
