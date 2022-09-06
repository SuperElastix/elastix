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

#ifndef itkParameterMapInterface_h
#define itkParameterMapInterface_h

#include "elxConversion.h"

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkMacro.h"
#include "itkNumericTraits.h"

#include "itkParameterFileParser.h"

#include <algorithm> // For count.
#include <iostream>
#include <memory>      // For unique_ptr.
#include <type_traits> // For is_same.

namespace itk
{

/** \class ParameterMapInterface
 *
 * \brief Implements functionality to get parameters from a parameter map.
 *
 * This class requires an std::map of parameter names and values, specified
 * as strings. Such a map can be created by the related class
 * itk::ParameterFileParser. This class implements functionality to get
 * parameters from this map and return them in the desired type. The function
 * that takes care of that is:\n
 *   ReadParameter( T parameterValue, std::string parameterName, ... )\n
 * which is templated over T. For convenience, several flavors of
 * ReadParameter() exist.
 *
 * The layout of ReadParameter is specified below.
 *
 * Warnings are created if the following two conditions are both satisfied:
 * 1) ReadParameter() is called with the function argument printWarningToStream
 *    set to true.
 * 2) The global member variable m_PrintErrorMessages is true.
 *
 * This class can be used in the following way:\n
 *
 * itk::ParameterMapInterface::Pointer p_interface = itk::ParameterMapInterface::New();
 * p_interface->SetParameterMap( parser->GetParameterMap() );
 * p_interface->PrintErrorMessages( true );
 * unsigned long parameterValue = 3;
 * unsigned int index = 2;
 * bool printWarning = true;
 * std::string errorMessage = "";
 * bool success = p_interface->ReadParameter( parameterValue,
 *   "ParameterName", index, printWarning, errorMessage );
 *
 *
 * Note that some of the templated functions are defined in the header to
 * get it compiling on some platforms.
 *
 * \sa itk::ParameterFileParser
 */

class ParameterMapInterface : public Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ParameterMapInterface);

  /** Standard ITK typedefs. */
  using Self = ParameterMapInterface;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ParameterMapInterface, Object);

  /** Typedefs. */
  using ParameterValuesType = ParameterFileParser::ParameterValuesType;
  using ParameterMapType = ParameterFileParser::ParameterMapType;

  /** Set the parameter map. */
  void
  SetParameterMap(const ParameterMapType & parMap);

  /** Option to print error and warning messages to a stream.
   * The default is true. If set to false no messages are printed.
   */
  // \todo: we could think of a warning level. (maybe you want warnings, but
  // not when for example a parameter is not found at entry entry_nr but at entry 0 instead
  itkSetMacro(PrintErrorMessages, bool);
  itkGetConstMacro(PrintErrorMessages, bool);

  /** Tells whether this parameter map has the parameter with the given name. */
  bool
  HasParameter(const std::string & parameterName) const
  {
    return this->m_ParameterMap.count(parameterName) > 0;
  }

  /** Get the number of entries for a given parameter. */
  std::size_t
  CountNumberOfParameterEntries(const std::string & parameterName) const;

  /** Get the desired parameter from the parameter map as type T.
   *
   * When requesting to read a parameter, multiple scenarios exist:
   * 1) The parameter is not found at all
   * 2) The parameter is found, but index entry_nr does not exist
   * 3) The parameter is found at the requested index, and cast is correct
   * 4) The parameter is found at the requested index, but the cast fails
   * What to return for these three options?
   * 1) -> return false + warning if desired
   * 2) -> return false + other warning if desired
   * 3) -> return true and no warning
   * 4) -> Throw exception: there is an error in the parameter file
   *
   */
  template <class T>
  bool
  ReadParameter(T &                 parameterValue,
                const std::string & parameterName,
                const unsigned int  entry_nr,
                const bool          printThisErrorMessage,
                std::string &       errorMessage) const
  {
    /** Reset the error message. */
    errorMessage = "";

    /** Get the number of entries. */
    std::size_t numberOfEntries = this->CountNumberOfParameterEntries(parameterName);

    /** Check if the requested parameter exists. */
    if (numberOfEntries == 0)
    {
      if (printThisErrorMessage && this->m_PrintErrorMessages)
      {
        std::ostringstream outputStringStream;
        outputStringStream << "WARNING: The parameter \"" << parameterName << "\", requested at entry number "
                           << entry_nr << ", does not exist at all.\n"
                           << "  The default value \"" << parameterValue << "\" is used instead.\n";
        errorMessage = outputStringStream.str();
      }

      return false;
    }

    /** Get the vector of parameters. */
    const ParameterValuesType & vec = this->m_ParameterMap.find(parameterName)->second;

    /** Check if it exists at the requested entry number. */
    if (entry_nr >= numberOfEntries)
    {
      if (printThisErrorMessage && this->m_PrintErrorMessages)
      {
        std::ostringstream outputStringStream;
        outputStringStream << "WARNING: The parameter \"" << parameterName << "\" does not exist at entry number "
                           << entry_nr << ".\n  The default value \"" << parameterValue << "\" is used instead.\n";
        errorMessage = outputStringStream.str();
      }
      return false;
    }

    /** Cast the string to type T. */
    bool castSuccesful = elastix::Conversion::StringToValue(vec[entry_nr], parameterValue);

    /** Check if the cast was successful. */
    if (!castSuccesful)
    {
      itkExceptionMacro("ERROR: Casting entry number "
                        << entry_nr << " for the parameter \"" << parameterName << "\" failed!\n"
                        << "  You tried to cast \"" << vec[entry_nr] << "\" from std::string to "
                        << typeid(parameterValue).name() << '\n');
    }

    return true;

  } // end ReadParameter()


  /** Boolean support. */
  bool
  ReadParameter(bool &              parameterValue,
                const std::string & parameterName,
                const unsigned int  entry_nr,
                const bool          printThisErrorMessage,
                std::string &       errorMessage) const;

  /** A shorter version of ReadParameter() that does not require the boolean
   * printThisErrorMessage. Instead the default value true is used.
   */
  template <class T>
  bool
  ReadParameter(T &                 parameterValue,
                const std::string & parameterName,
                const unsigned int  entry_nr,
                std::string &       errorMessage) const
  {
    return this->ReadParameter(parameterValue, parameterName, entry_nr, true, errorMessage);
  }


  /** An extended version of ReadParameter() that takes prefixes and
   * default entry numbers (for convenience).
   * This function tries to read parameterName, but also prefix+parameterName.
   * Also, multiple entries are tried, entry_nr as well as default_entry_nr.
   */
  template <class T>
  bool
  ReadParameter(T &                 parameterValue,
                const std::string & parameterName,
                const std::string & prefix,
                const unsigned int  entry_nr,
                const int           default_entry_nr,
                const bool          printThisErrorMessage,
                std::string &       errorMessage) const
  {
    std::string fullname = prefix + parameterName;
    bool        found = false;

    /** Silently try to read the parameter. */
    std::string dummyString = "";
    if (default_entry_nr >= 0)
    {
      /** Try the default_entry_nr if the entry_nr is not found. */
      unsigned int uintdefault = static_cast<unsigned int>(default_entry_nr);
      found |= this->ReadParameter(parameterValue, parameterName, uintdefault, false, dummyString);
      found |= this->ReadParameter(parameterValue, parameterName, entry_nr, false, dummyString);
      found |= this->ReadParameter(parameterValue, fullname, uintdefault, false, dummyString);
      found |= this->ReadParameter(parameterValue, fullname, entry_nr, false, dummyString);
    }
    else
    {
      /** Just try the entry_nr. */
      found |= this->ReadParameter(parameterValue, parameterName, entry_nr, false, dummyString);
      found |= this->ReadParameter(parameterValue, fullname, entry_nr, false, dummyString);
    }

    /** If we haven't found anything, give a warning that the default value
     * provided by the caller is used.
     */
    if (!found && printThisErrorMessage && this->m_PrintErrorMessages)
    {
      return this->ReadParameter(parameterValue, parameterName, entry_nr, true, errorMessage);
    }

    return found;
  }


  /** A shorter version of the extended ReadParameter() that does not require
   * the boolean printThisErrorMessage. Instead the default value true is used.
   */
  template <class T>
  bool
  ReadParameter(T &                 parameterValue,
                const std::string & parameterName,
                const std::string & prefix,
                const unsigned int  entry_nr,
                const unsigned int  default_entry_nr,
                std::string &       errorMessage) const
  {
    return this->ReadParameter(parameterValue, parameterName, prefix, entry_nr, default_entry_nr, true, errorMessage);
  }


  /** An extended version that reads all parameters in a range at once. */
  template <class T>
  bool
  ReadParameter(std::vector<T> &    parameterValues,
                const std::string & parameterName,
                const unsigned int  entry_nr_start,
                const unsigned int  entry_nr_end,
                const bool          printThisErrorMessage,
                std::string &       errorMessage) const
  {
    /** Reset the error message. */
    errorMessage = "";

    /** Get the number of entries. */
    std::size_t numberOfEntries = this->CountNumberOfParameterEntries(parameterName);

    /** Check if the requested parameter exists. */
    if (numberOfEntries == 0)
    {
      if (printThisErrorMessage && this->m_PrintErrorMessages)
      {
        std::ostringstream outputStringStream;
        outputStringStream << "WARNING: The parameter \"" << parameterName << "\", requested between entry numbers "
                           << entry_nr_start << " and " << entry_nr_end << ", does not exist at all.\n"
                           << "  The default values are used instead.\n";
        errorMessage = outputStringStream.str();
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
                                                    << entry_nr_end << ".\nThe default value \""
                                                    << itk::NumericTraits<T>::Zero << "\" is used instead.\n");
    }

    /** Get the vector of parameters. */
    const ParameterValuesType & vec = this->m_ParameterMap.find(parameterName)->second;

    /** The default is filled with zero's.
    parameterValues.clear();
    parameterValues.resize( entry_nr_end - entry_nr_start + 1,
      itk::NumericTraits<T>::ZeroValue() );
    */

    /** Get all parameters at once. */
    unsigned int j = 0;
    for (unsigned int i = entry_nr_start; i < entry_nr_end + 1; ++i)
    {
      /** Cast the string to type T. */
      bool castSuccesful = elastix::Conversion::StringToValue(vec[i], parameterValues[j]);
      ++j;

      /** Check if the cast was successful. */
      if (!castSuccesful)
      {
        itkExceptionMacro("ERROR: Casting entry number "
                          << i << " for the parameter \"" << parameterName << "\" failed!\n"
                          << "  You tried to cast \"" << vec[i] << "\" from std::string to "
                          << typeid(parameterValues[0]).name() << '\n');
      }
    }

    return true;
  }


  /** Provide a specialization for std::string, for efficiency. */
  bool
  ReadParameter(std::vector<std::string> & parameterValues,
                const std::string &        parameterName,
                const unsigned int         entry_nr_start,
                const unsigned int         entry_nr_end,
                const bool                 printThisErrorMessage,
                std::string &              errorMessage) const;


  /** Returns the values of the specified parameter. */
  std::vector<std::string>
  GetValues(const std::string & parameterName) const
  {
    const auto found = m_ParameterMap.find(parameterName);
    return (found == m_ParameterMap.cend()) ? std::vector<std::string>{} : found->second;
  }

  /** Retrieves the values of the specified parameter. Returns null when the
   * map does not contain the specified parameter. Throws an exception when
   * it fails to convert each of the values to the specified type `T`.
   */
  template <typename T>
  std::unique_ptr<std::vector<T>>
  RetrieveValues(const std::string & parameterName) const
  {
    const auto found = m_ParameterMap.find(parameterName);
    if (found == m_ParameterMap.end())
    {
      return nullptr;
    }
    std::vector<T> result;
    result.reserve(found->second.size());

    for (const std::string & str : found->second)
    {
      T value{};

      if (elastix::Conversion::StringToValue(str, value))
      {
        result.push_back(value);
      }
      else
      {
        const auto entry_nr = &str - found->second.data();
        itkExceptionMacro(<< "Failed to cast parameter \"" << parameterName << "\" entry number " << entry_nr
                          << " value \"" << str << "\" to type \"" << typeid(T).name() << "\"!");
      }
    }
    return std::make_unique<std::vector<T>>(std::move(result));
  }


protected:
  ParameterMapInterface();
  ~ParameterMapInterface() override;

private:
  /** Member variable to store the parameters. */
  ParameterMapType m_ParameterMap;

  bool m_PrintErrorMessages{ true };
};

} // end of namespace itk

#endif // end itkParameterMapInterface_h
