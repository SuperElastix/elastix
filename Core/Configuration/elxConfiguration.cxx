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

#include "elxConfiguration.h"

#include "elxConversion.h"
#include "elxTransformIO.h"
#include "elxTransformFactoryRegistration.h"

namespace elastix
{
namespace
{
itk::ParameterFileParser::ParameterMapType
AddDataFromExternalTransformFile(const std::string &                        parameterFileName,
                                 itk::ParameterFileParser::ParameterMapType parameterMap)
{
  const auto endOfParameterMap = parameterMap.end();
  const auto transformParameter = parameterMap.find("Transform");

  if ((transformParameter != endOfParameterMap) &&
      (transformParameter->second == itk::ParameterFileParser::ParameterValuesType{ "File" }))
  {
    const auto transformFileNameParameter = parameterMap.find("TransformFileName");

    if ((transformFileNameParameter != endOfParameterMap) && (transformFileNameParameter->second.size() == 1))
    {
      const auto & transformFileName = transformFileNameParameter->second.front();

      if (!transformFileName.empty())
      {
        const bool transformFileNameSpecifiesFullPath = (transformFileName.front() == '/') ||
                                                        (transformFileName.front() == '\\') ||
                                                        (transformFileName.find(':') != std::string::npos);

        const auto getDirectoryPath = [](const std::string & fileName) {
          const auto foundPosition = fileName.find_last_of("/\\");
          return (foundPosition == std::string::npos) ? std::string() : std::string(fileName, 0, foundPosition + 1);
        };

        const auto itkTransform = TransformIO::Read(
          (transformFileNameSpecifiesFullPath ? "" : getDirectoryPath(parameterFileName)) + transformFileName);

        if (itkTransform != nullptr)
        {
          transformParameter->second = { TransformIO::ConvertITKNameOfClassToElastixClassName(
            itkTransform->GetNameOfClass()) };
          parameterMap["ITKTransformParameters"] = Conversion::ToVectorOfStrings(itkTransform->GetParameters());
          parameterMap["ITKTransformFixedParameters"] =
            Conversion::ToVectorOfStrings(itkTransform->GetFixedParameters());
          parameterMap["ITKTransformType"] = { itkTransform->GetTransformTypeAsString() };
        }
      }
    }
  }
  return parameterMap;
}

} // namespace
/**
 * ********************* Constructor ****************************
 */

Configuration::Configuration() = default;


/**
 * ******************** PrintParameterFile ***************************
 */

void
Configuration::PrintParameterFile() const
{
  /** Read what's in the parameter file. */
  std::string params = m_ParameterFileParser->ReturnParameterFileAsString();

  /** Separate clearly in log-file, before and after writing the parameter file. */
  log::info_to_log_file(std::ostringstream{} << '\n'
                                             << "=============== start of ParameterFile: "
                                             << this->GetParameterFileName() << " ===============\n"
                                             << params << '\n'
                                             << "=============== end of ParameterFile: " << this->GetParameterFileName()
                                             << " ===============\n");

} // end PrintParameterFile()


/**
 * ************************ BeforeAll ***************************
 */

int
Configuration::BeforeAll()
{
  if (!BaseComponent::IsElastixLibrary())
  {
    this->PrintParameterFile();
  }
  return 0;

} // end BeforeAll()


/**
 * ************************ BeforeAllTransformix ***************************
 */

int
Configuration::BeforeAllTransformix()
{
  this->PrintParameterFile();
  return 0;

} // end BeforeAllTransformix()


/**
 * ********************** Initialize ****************************
 */

int
Configuration::Initialize(const CommandLineArgumentMapType & _arg)
{
  TransformFactoryRegistration::RegisterTransforms();

  /** The first part is getting the command line arguments and setting them
   * in the configuration. From the command line arguments we find the name
   * of the parameter text file. The second part is then to get and set the
   * parameter in this configuration.
   */

  /** Store the command line arguments. */
  m_CommandLineArgumentMap = _arg;

  /** This function can either be called by elastix or transformix.
   * If called by elastix the command line argument "-p" has to be
   * specified. If called by transformix the command line argument
   * "-tp" has to be specified.
   * NOTE: this implies that one can not use "-tp" for elastix and
   * "-p" for transformix.
   */
  std::string p = this->GetCommandLineArgument("-p");
  std::string tp = this->GetCommandLineArgument("-tp");

  if (!p.empty() && tp.empty())
  {
    /** elastix called Initialize(). */
    this->SetParameterFileName(p.c_str());
  }
  else if (p.empty() && !tp.empty())
  {
    /** transformix called Initialize(). */
    this->SetParameterFileName(tp.c_str());
  }
  else if (p.empty() && tp.empty())
  {
    log::error(std::ostringstream{} << "ERROR: No (Transform-)Parameter file has been entered\n"
                                    << "for elastix: command line option \"-p\"\n"
                                    << "for transformix: command line option \"-tp\"");
    return 1;
  }
  else
  {
    /** Both "p" and "tp" are used, which is prohibited. */
    log::error(std::ostringstream{} << "ERROR: Both \"-p\" and \"-tp\" are used, which is prohibited.");
    return 1;
  }

  /** Read the ParameterFile. */
  m_ParameterFileParser->SetParameterFileName(m_ParameterFileName);
  try
  {
    log::info("Reading the elastix parameters from file ...\n");
    m_ParameterFileParser->ReadParameterFile();
  }
  catch (const itk::ExceptionObject & excp)
  {
    log::error(std::ostringstream{} << "ERROR: when reading the parameter file:\n" << excp);
    return 1;
  }

  /** Connect the parameter file reader to the interface. */
  m_ParameterMapInterface->SetParameterMap(
    AddDataFromExternalTransformFile(m_ParameterFileName, m_ParameterFileParser->GetParameterMap()));

  /** Silently check in the parameter file if error messages should be printed. */
  m_ParameterMapInterface->SetPrintErrorMessages(false);
  bool printErrorMessages = true;
  this->ReadParameter(printErrorMessages, "PrintErrorMessages", 0, false);
  m_ParameterMapInterface->SetPrintErrorMessages(printErrorMessages);

  /** Set the initialized flag. */
  m_IsInitialized = true;

  /** Return a value.*/
  return 0;

} // end Initialize()


/**
 * ********************** Initialize ****************************
 */

int
Configuration::Initialize(const CommandLineArgumentMapType &                 _arg,
                          const itk::ParameterFileParser::ParameterMapType & inputMap)
{
  TransformFactoryRegistration::RegisterTransforms();

  /** The first part is getting the command line arguments and setting them
   * in the configuration. From the command line arguments we find the name
   * of the parameter text file. The second part is then to get and set the
   * parameter in this configuration.
   */

  /** Store the command line arguments. */
  m_CommandLineArgumentMap = _arg;

  m_ParameterMapInterface->SetParameterMap(AddDataFromExternalTransformFile(m_ParameterFileName, inputMap));

  /** Silently check in the parameter file if error messages should be printed. */
  m_ParameterMapInterface->SetPrintErrorMessages(false);
  bool printErrorMessages = true;
  this->ReadParameter(printErrorMessages, "PrintErrorMessages", 0, false);
  m_ParameterMapInterface->SetPrintErrorMessages(printErrorMessages);

  /** Set the initialized flag. */
  m_IsInitialized = true;

  /** Return a value.*/
  return 0;

} // end Initialize()


/**
 * ********************** IsInitialized ***************************
 */

bool
Configuration::IsInitialized() const
{
  return m_IsInitialized;

} // end IsInitialized()


/**
 * ****************** GetCommandLineArgument ********************
 */

std::string
Configuration::GetCommandLineArgument(const std::string & key) const
{
  const auto found = m_CommandLineArgumentMap.find(key);

  /** Check if the argument was given. If no return "". */
  if (found == m_CommandLineArgumentMap.end())
  {
    return "";
  }

  return found->second;

} // end GetCommandLineArgument()


/**
 * ****************** SetCommandLineArgument ********************
 */

void
Configuration::SetCommandLineArgument(const std::string & key, const std::string & value)
{
  m_CommandLineArgumentMap[key] = value;

} // end SetCommandLineArgument()


} // end namespace elastix
