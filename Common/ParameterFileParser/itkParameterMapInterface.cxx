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

#include "itkParameterMapInterface.h"

namespace itk
{

/**
 * **************** Constructor ***************
 */

ParameterMapInterface::ParameterMapInterface() = default;


/**
 * **************** Destructor ***************
 */

ParameterMapInterface ::~ParameterMapInterface() = default;


/**
 * **************** SetParameterMap ***************
 */

void
ParameterMapInterface::SetParameterMap(const ParameterMapType & parMap)
{
  if (!parMap.empty())
  {
    this->m_ParameterMap = parMap;
  }

} // end SetParameterMap()


/**
 * **************** CountNumberOfParameterEntries ***************
 */

std::size_t
ParameterMapInterface::CountNumberOfParameterEntries(const std::string & parameterName) const
{
  if (this->m_ParameterMap.count(parameterName))
  {
    return this->m_ParameterMap.find(parameterName)->second.size();
  }
  return 0;

} // end CountNumberOfParameterEntries()


/**
 * **************** ReadParameter ***************
 */

bool
ParameterMapInterface::ReadParameter(bool &              parameterValue,
                                     const std::string & parameterName,
                                     const unsigned int  entry_nr,
                                     const bool          produceWarningMessage,
                                     std::string &       warningMessage) const
{
  /** Translate the default boolean to string. */
  std::string parameterValueString;
  if (parameterValue)
  {
    parameterValueString = "true";
  }
  else
  {
    parameterValueString = "false";
  }

  /** Read the boolean as a string. */
  bool dummy =
    this->ReadParameter(parameterValueString, parameterName, entry_nr, produceWarningMessage, warningMessage);

  /** Translate the read-in string to boolean. */
  parameterValue = false;

  if (!elastix::Conversion::StringToValue(parameterValueString, parameterValue))
  {
    /** Trying to read a string other than "true" or "false" as a boolean. */
    itkExceptionMacro("ERROR: Entry number "
                      << entry_nr << " for the parameter \"" << parameterName
                      << "\" should be a boolean, i.e. either \"true\" or \"false\", but it reads \""
                      << parameterValueString << "\".");
  }

  return dummy;

} // end ReadParameter()


/**
 * **************** ReadParameter ***************
 */

bool
ParameterMapInterface::ReadParameter(std::vector<std::string> & parameterValues,
                                     const std::string &        parameterName,
                                     const unsigned int         entry_nr_start,
                                     const unsigned int         entry_nr_end,
                                     const bool                 produceWarningMessage,
                                     std::string &              warningMessage) const
{
  /** Reset the warning message. */
  warningMessage = "";

  /** Get the number of entries. */
  std::size_t numberOfEntries = this->CountNumberOfParameterEntries(parameterName);

  /** Check if the requested parameter exists. */
  if (numberOfEntries == 0)
  {
    if (produceWarningMessage && this->m_PrintErrorMessages)
    {
      std::ostringstream outputStringStream;
      outputStringStream << "WARNING: The parameter \"" << parameterName << "\", requested between entry numbers "
                         << entry_nr_start << " and " << entry_nr_end << ", does not exist at all.\n"
                         << "  The default values are used instead.";
      warningMessage = outputStringStream.str();
    }
    return false;
  }

  /** Check. */
  if (entry_nr_start > entry_nr_end)
  {
    /** Programming error: just throw an exception. */
    itkExceptionMacro("WARNING: The entry number start ("
                      << entry_nr_start << ") should be smaller than entry number end (" << entry_nr_end
                      << "). It was requested for parameter \"" << parameterName << "\".\n");
  }

  /** Check if it exists at the requested entry numbers. */
  if (entry_nr_end >= numberOfEntries)
  {
    itkExceptionMacro("WARNING: The parameter \"" << parameterName << "\" does not exist at entry number "
                                                  << entry_nr_end
                                                  << ".\nThe default empty string \"\" is used instead.\n");
  }

  /** Get the vector of parameters. */
  const ParameterValuesType & vec = this->m_ParameterMap.find(parameterName)->second;

  /** Copy all parameters at once. */
  std::vector<std::string>::const_iterator it = vec.begin();
  parameterValues.clear();
  parameterValues.assign(it + entry_nr_start, it + entry_nr_end + 1);

  return true;
}


} // end namespace itk
