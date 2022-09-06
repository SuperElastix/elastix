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
#ifndef elxConfiguration_h
#define elxConfiguration_h

#include "itkObject.h"
#include "elxBaseComponent.h"

#include "itkParameterFileParser.h"
#include "itkParameterMapInterface.h"
#include <map>
#include "xoutmain.h"

namespace elastix
{

/**
 * \class Configuration
 * \brief A class that deals with user given parameters and command line arguments.
 *
 * The Configuration class provides the functions
 * ReadParameter() (to read parameters from the parameter file) and
 * ReadCommandLineArgument(), and provides an easy way to get the
 * current elastix level.
 *
 * \parameter PrintErrorMessages: defines if warnings and errors should be
 * printed to screen, e.g. when a parameter cannot be found and the default
 * is used.
 * example: <tt>(PrintErrorMessages "false")</tt>\n
 * Default: "true"
 *
 * \ingroup Configuration
 */

class Configuration
  : public itk::Object
  , public BaseComponent
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(Configuration);

  /** Standard itk.*/
  using Self = Configuration;
  using Superclass1 = itk::Object;
  using Superclass2 = BaseComponent;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Standard part of all itk objects. */
  itkTypeMacro(Configuration, itk::Object);

  /** Typedefs for the command line arguments. */
  using CommandLineArgumentMapType = std::map<std::string, std::string>;
  using CommandLineEntryType = CommandLineArgumentMapType::value_type;

  /** Get and Set CommandLine arguments into the argument map. */
  std::string
  GetCommandLineArgument(const std::string & key) const;

  void
  SetCommandLineArgument(const std::string & key, const std::string & value);

  /** Get/Set the name of the parameterFileName. */
  itkGetStringMacro(ParameterFileName);
  itkSetStringMacro(ParameterFileName);

  /** Pass the command line arguments as a map.
   * One of the command line arguments should be either:
   * -p \<parameter_file\>         or
   * -tp \<transform_parameter_file\>.
   * The specified (transform) parameter file is read by the
   * itk::ParameterFileParser and passed to the itk::ParameterMapInterface.
   */
  virtual int
  Initialize(const CommandLineArgumentMapType & _arg);

  virtual int
  Initialize(const CommandLineArgumentMapType & _arg, const itk::ParameterFileParser::ParameterMapType & inputMap);

  /** True, if Initialize was successfully called. */
  virtual bool
  IsInitialized() const; // to elxconfigurationbase

  /** Other elastix related information. */

  /** Get and Set the elastix level. */
  itkSetMacro(ElastixLevel, unsigned int);
  itkGetConstMacro(ElastixLevel, unsigned int);

  /** Get and Set the total number of elastix levels. */
  itkSetMacro(TotalNumberOfElastixLevels, unsigned int);
  itkGetConstMacro(TotalNumberOfElastixLevels, unsigned int);

  /***/
  virtual bool
  GetPrintErrorMessages()
  {
    return this->m_ParameterMapInterface->GetPrintErrorMessages();
  }


  /** Set/Get whether warnings are allowed to be printed, when reading a parameter. *
  itkSetMacro( Silent, bool );
  itkGetConstMacro( Silent, bool );
  */

  /** Methods that is called at the very beginning of elastixTemplate::Run.
   * \li Prints the parameter file
   */
  int
  BeforeAll() override;

  /** Methods that is called at the very beginning of elastixTemplate::ApplyTransform.
   * \li Prints the parameter file
   */
  virtual int
  BeforeAllTransformix();

  /** Interface to the ParameterMapInterface. */

  /** Count the number of parameters. */
  std::size_t
  CountNumberOfParameterEntries(const std::string & parameterName) const
  {
    return this->m_ParameterMapInterface->CountNumberOfParameterEntries(parameterName);
  }


  /** Read a parameter from the parameter file. */
  template <class T>
  bool
  ReadParameter(T &                 parameterValue,
                const std::string & parameterName,
                const unsigned int  entry_nr,
                const bool          printThisErrorMessage) const
  {
    std::string errorMessage = "";
    bool        found = this->m_ParameterMapInterface->ReadParameter(
      parameterValue, parameterName, entry_nr, printThisErrorMessage, errorMessage);
    if (!errorMessage.empty())
    {
      xl::xout["error"] << errorMessage;
    }

    return found;
  }


  /** Read a parameter from the parameter file. */
  template <class T>
  bool
  ReadParameter(T & parameterValue, const std::string & parameterName, const unsigned int entry_nr) const
  {
    std::string errorMessage = "";
    bool found = this->m_ParameterMapInterface->ReadParameter(parameterValue, parameterName, entry_nr, errorMessage);
    if (!errorMessage.empty())
    {
      xl::xout["error"] << errorMessage;
    }

    return found;
  }


  /** Read a parameter from the parameter file. */
  template <class T>
  bool
  ReadParameter(T &                 parameterValue,
                const std::string & parameterName,
                const std::string & prefix,
                const unsigned int  entry_nr,
                const int           default_entry_nr,
                const bool          printThisErrorMessage) const
  {
    std::string errorMessage = "";
    bool        found = this->m_ParameterMapInterface->ReadParameter(
      parameterValue, parameterName, prefix, entry_nr, default_entry_nr, printThisErrorMessage, errorMessage);
    if (!errorMessage.empty())
    {
      xl::xout["error"] << errorMessage;
    }

    return found;
  }


  /** Read a parameter from the parameter file. */
  template <class T>
  bool
  ReadParameter(T &                 parameterValue,
                const std::string & parameterName,
                const std::string & prefix,
                const unsigned int  entry_nr,
                const int           default_entry_nr) const
  {
    std::string errorMessage = "";
    bool        found = this->m_ParameterMapInterface->ReadParameter(
      parameterValue, parameterName, prefix, entry_nr, default_entry_nr, errorMessage);
    if (!errorMessage.empty())
    {
      xl::xout["error"] << errorMessage;
    }

    return found;
  }


  /** Tells whether its parameter map has the parameter with the given name. */
  bool
  HasParameter(const std::string & parameterName) const
  {
    return m_ParameterMapInterface->HasParameter(parameterName);
  }


  /** Returns the values of the specified parameter (from the parameter file). */
  std::vector<std::string>
  GetValuesOfParameter(const std::string & parameterName) const
  {
    return m_ParameterMapInterface->GetValues(parameterName);
  }


  /** Retrieves the values of the specified parameter (from the parameter file).
   * Returns null when the parameter map does not contain the specified
   * parameter. Throws an exception when it fails to convert each of the
   * parameter values to the specified type `T`.
   */
  template <typename T>
  std::unique_ptr<std::vector<T>>
  RetrieveValuesOfParameter(const std::string & parameterName) const
  {
    return m_ParameterMapInterface->RetrieveValues<T>(parameterName);
  }


  /** Read a range of parameters from the parameter file. */
  template <class T>
  bool
  ReadParameter(std::vector<T> &    parameterValues,
                const std::string & parameterName,
                const unsigned int  entry_nr_start,
                const unsigned int  entry_nr_end,
                const bool          printThisErrorMessage) const
  {
    std::string errorMessage = "";
    bool        found = this->m_ParameterMapInterface->ReadParameter(
      parameterValues, parameterName, entry_nr_start, entry_nr_end, printThisErrorMessage, errorMessage);
    if (!errorMessage.empty())
    {
      xl::xout["error"] << errorMessage;
    }

    return found;
  }


protected:
  Configuration();
  ~Configuration() override = default;

  /** Print the parameter file to the log file. Called by BeforeAll().
   * This function is not really generic. It's just added because it needs to be
   * called by both BeforeAll and BeforeAllTransformix.
   */
  virtual void
  PrintParameterFile() const;

private:
  CommandLineArgumentMapType          m_CommandLineArgumentMap{};
  std::string                         m_ParameterFileName{};
  itk::ParameterFileParser::Pointer   m_ParameterFileParser{ itk::ParameterFileParser::New() };
  itk::ParameterMapInterface::Pointer m_ParameterMapInterface{ itk::ParameterMapInterface::New() };

  bool         m_IsInitialized{ false };
  unsigned int m_ElastixLevel{ 0 };
  unsigned int m_TotalNumberOfElastixLevels{ 1 };
};

} // end namespace elastix

#endif // end #ifndef elxConfiguration_h
