/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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
  if( !parMap.empty() )
  {
    this->m_ParameterMap = parMap;
  }

} // end SetParameterMap()


/**
 * **************** CountNumberOfParameterEntries ***************
 */

std::size_t
ParameterMapInterface
::CountNumberOfParameterEntries(
  const std::string & parameterName ) const
{
  if( this->m_ParameterMap.count( parameterName ) )
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
  if( parameterValue )
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
  if( parameterValueString == "true" )
  {
    parameterValue = true;
  }
  else if( parameterValueString == "false" )
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


/**
 * **************** ReadParameter ***************
 */

bool
ParameterMapInterface
::ReadParameter(
  std::vector< std::string > & parameterValues,
  const std::string & parameterName,
  const unsigned int entry_nr_start,
  const unsigned int entry_nr_end,
  const bool printThisErrorMessage,
  std::string & errorMessage ) const
{
  /** Reset the error message. */
  errorMessage = "";

  /** Get the number of entries. */
  std::size_t numberOfEntries = this->CountNumberOfParameterEntries(
    parameterName );

  /** Check if the requested parameter exists. */
  if( numberOfEntries == 0 )
  {
    std::stringstream ss;
    ss << "WARNING: The parameter \"" << parameterName
       << "\", requested between entry numbers " << entry_nr_start
       << " and " << entry_nr_end
       << ", does not exist at all.\n"
       << "  The default values are used instead." << std::endl;
    if( printThisErrorMessage && this->m_PrintErrorMessages )
    {
      errorMessage = ss.str();
    }
    return false;
  }

  /** Check. */
  if( entry_nr_start > entry_nr_end )
  {
    std::stringstream ss;
    ss << "WARNING: The entry number start (" << entry_nr_start
       << ") should be smaller than entry number end (" << entry_nr_end
       << "). It was requested for parameter \"" << parameterName
       << "\"." << std::endl;

    /** Programming error: just throw an exception. */
    itkExceptionMacro( << ss.str() );
  }

  /** Check if it exists at the requested entry numbers. */
  if( entry_nr_end >= numberOfEntries )
  {
    std::stringstream ss;
    ss << "WARNING: The parameter \"" << parameterName
       << "\" does not exist at entry number " << entry_nr_end
       << ".\nThe default empty string \"\" is used instead." << std::endl;
    itkExceptionMacro( << ss.str() );
  }

  /** Get the vector of parameters. */
  const ParameterValuesType & vec = this->m_ParameterMap.find( parameterName )->second;

  /** Copy all parameters at once. */
  std::vector< std::string >::const_iterator it = vec.begin();
  parameterValues.clear();
  parameterValues.assign( it + entry_nr_start, it + entry_nr_end + 1 );

  return true;
}


} // end namespace itk

#endif // end __itkParameterMapInterface_cxx
