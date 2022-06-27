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

// Elastix header files:
#include "elastix.h" // For GetCurrentDateAndTime()
#include "elxConversion.h"
#include "elxDefaultConstruct.h"
#include "elxForEachSupportedImageType.h"
#include "elxMainExeUtilities.h"
#include "elxTransformixMain.h"
#include <Core/elxVersionMacros.h>
#include "itkUseMevisDicomTiff.h"
#include "itkTransformixFilter.h"
#include "itkImageFileCastWriter.h"

// ITK header files:
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTimeProbe.h>
#include <itksys/SystemInformation.hxx>
#include <itksys/SystemTools.hxx>

// Standard C++ header files:
#include <cassert>
#include <climits> // For UINT_MAX.
#include <cstddef> // For size_t.
#include <deque>
#include <iostream>
#include <limits>
#include <utility> // For index_sequence.
#include <vector>


constexpr const char * transformixHelpText =
  /* Print the version. */
  "transformix version: " ELASTIX_VERSION_STRING "\n\n"

  /* What is transformix? */
  "transformix applies a transform on an input image and/or "
  "generates a deformation field.\n"
  "The transform is specified in the transform-parameter file.\n"
  "  --help, -h displays this message and exit\n"
  "  --version  output version information and exit\n\n"

  /* Mandatory arguments. */
  "Call transformix from the command line with mandatory arguments:\n"
  "  -out      output directory\n"
  "  -tp       transform-parameter file, only 1\n\n"

  /* Optional arguments. */
  "Optional extra commands:\n"
  "  -in       input image to deform\n"
  "  -def      file containing input-image points; the point are transformed\n"
  "            according to the specified transform-parameter file\n"
  "            use \"-def all\" to transform all points from the input-image, which\n"
  "            effectively generates a deformation field.\n"
  "  -jac      use \"-jac all\" to generate an image with the determinant of the\n"
  "            spatial Jacobian\n"
  "  -jacmat   use \"-jacmat all\" to generate an image with the spatial Jacobian\n"
  "            matrix at each voxel\n"
  "  -priority set the process priority to high, abovenormal, normal (default),\n"
  "            belownormal, or idle (Windows only option)\n"
  "  -threads  set the maximum number of threads of transformix\n"
  "\nAt least one of the options \"-in\", \"-def\", \"-jac\", or \"-jacmat\" should be given.\n\n"

  /* The parameter file. */
  "The transform-parameter file must contain all the information "
  "necessary for transformix to run properly. That includes which transform "
  "to use, with which parameters, etc. For a usable transform-parameter file, "
  "run elastix, and inspect the output file \"TransformParameters.0.txt\".\n\n"

  "Need further help? Please check:\n"
  " * the elastix website: https://elastix.lumc.nl\n"
  " * the source code repository site: https://github.com/SuperElastix/elastix\n"
  " * the discussion forum: https://groups.google.com/g/elastix-imageregistration";

namespace
{
using ParameterValuesType = std::vector<std::string>;
using ParameterMapType = std::map<std::string, ParameterValuesType>;

template <typename T>
T
GetSingleParameterValue(const ParameterMapType & parameterMap,
                        const std::string &      parameterName,
                        const T &                defaultValue)
{
  const auto found = parameterMap.find(parameterName);

  if (found == parameterMap.end())
  {
    return defaultValue;
  }
  const auto & parameterValues = found->second;

  if (parameterValues.size() != 1)
  {
    itkGenericExceptionMacro("The parameter named \"" << parameterName << "\" has " << parameterValues.size()
                                                      << " values. It must have exactly one parameter value!");
  }

  const auto & str = parameterValues.front();

  if (str.empty())
  {
    itkGenericExceptionMacro("The parameter named \"" << parameterName << "\" has an empty value!");
  }

  T value;
  if (elx::Conversion::StringToValue(str, value))
  {
    return value;
  }
  itkGenericExceptionMacro("Failed to convert the value \"" << str << " of the parameter named \"" << parameterName
                                                            << " to type '" << typeid(T).name() << "'");
}
} // namespace

int
main(int argc, char ** argv)
{
  try
  {
    assert(elastix::BaseComponent::IsElastixLibrary());

    // Check if "-help" or "--version" was asked for.
    if (argc == 1)
    {
      std::cout << "Use \"transformix --help\" for information about transformix-usage." << std::endl;
      return 0;
    }
    else if (argc == 2)
    {
      const std::string argument(argv[1]);
      if (argument == "-help" || argument == "--help" || argument == "-h")
      {
        std::cout << transformixHelpText << std::endl;
        return 0;
      }
      else if (argument == "--version")
      {
        std::cout << "transformix version: " ELASTIX_VERSION_STRING << std::endl;
        return 0;
      }
      else if (argument == "--extended-version")
      {
        std::cout << elx::GetExtendedVersionInformation("transformix");
        return 0;
      }
      else
      {
        std::cout << "Use \"transformix --help\" for information about transformix-usage." << std::endl;
        return 0;
      }
    }

    // Support Mevis Dicom Tiff (if selected in cmake) */
    RegisterMevisDicomTiff();

    const auto argMap = [argc, argv] {
      std::map<std::string, std::string> argMap;
      // Put the command line parameters into a map.
      for (int i = 1; i < (argc - 1); i += 2)
      {
        const char * const key = argv[i];
        const char * const value = argv[i + 1];

        // Attempt to save the arguments in the ArgumentMap.
        if (!argMap.insert(std::pair<std::string, std::string>(key, value)).second)
        {
          // Duplicate arguments.
          std::cerr << "WARNING!\nArgument " << key << "is only required once.\nArguments " << key << " " << value
                    << "are ignored\n";
        }
      }
      return argMap;
    }();

    const auto outFolderFound = argMap.find("-out");

    if (outFolderFound == argMap.cend())
    {
      std::cerr << "ERROR: No CommandLine option \"-out\" given!\n";
      return -2;
    }

    const auto addSlash =
      [](const std::string & value) { // Make sure that last character of the output folder equals a '/' or '\'.
        const char last = value.back();
        if (last != '/' && last != '\\')
        {
          return value + '/';
        }
        return value;
      };

    const std::string outFolder = elx::Conversion::ToNativePathNameSeparators(addSlash(outFolderFound->second));

    int returndummy = 0;

    /** Check that the option "-tp" is given. */
    const auto tpArg = argMap.find("-tp");
    if (tpArg == argMap.cend())
    {
      std::cerr << "ERROR: No CommandLine option \"-tp\" given!\n";
      returndummy |= -1;
    }

    // Check that at least one of the following options is given.
    constexpr const char * options[] = { "-in", "-ipp", "-def", "-jac", "-jacmat" };

    if (std::all_of(std::begin(options), std::end(options), [&argMap](const char * const option) {
          return argMap.count(option) == 0;
        }))
    {
      std::cerr
        << "ERROR: At least one of the CommandLine options \"-in\", \"-def\", \"-jac\", or \"-jacmat\" should be "
           "given!\n";
      returndummy |= -1;
    }

    // Check if the output directory exists.
    if (!itksys::SystemTools::FileIsDirectory(outFolder))
    {
      std::cerr << "ERROR: the output directory \"" << outFolder
                << "\" does not exist.\nYou are responsible for creating it.\n";
      returndummy |= -2;
    }
    else
    {
      /** Setup xout. */
      const std::string logFileName = outFolder + "transformix.log";
      int               returndummy2 = elx::xoutSetup(logFileName.c_str(), true, true);
      if (returndummy2)
      {
        std::cerr << "ERROR while setting up xout.\n";
      }
      returndummy |= returndummy2;
    }

    /** Stop if some fatal errors occurred. */
    if (returndummy != EXIT_SUCCESS)
    {
      return returndummy;
    }

    elxout << std::endl;

    /** Declare a timer, start it and print the start time. */
    itk::TimeProbe totaltimer;
    totaltimer.Start();
    elxout << "transformix is started at " << GetCurrentDateAndTime() << ".\n" << std::endl;

    // Print where transformix was run.
    elxout << "which transformix:   " << argv[0] << '\n' << elx::GetExtendedVersionInformation("transformix", "  ");
    elx::PrintArguments(elxout, argv);

    itksys::SystemInformation info;
    info.RunCPUCheck();
    info.RunOSCheck();
    info.RunMemoryCheck();
    elxout << "transformix runs at: " << info.GetHostname() << "\n  " << info.GetOSName() << " " << info.GetOSRelease()
           << (info.Is64Bits() ? " (x64), " : ", ") << info.GetOSVersion() << "\n  with "
           << info.GetTotalPhysicalMemory() << " MB memory, and " << info.GetNumberOfPhysicalCPU() << " cores @ "
           << static_cast<unsigned int>(info.GetProcessorClockFrequency()) << " MHz." << std::endl;

    /**
     * ********************* START TRANSFORMATION *******************
     */

    const auto transformParameterFile = tpArg->second;

    const auto transformParameterMap = itk::ParameterFileParser::ReadParameterMap(transformParameterFile);

    const auto movingImageDimension = transformParameterMap.at("MovingImageDimension").at(0);
    const auto foundResultImagePixelType = transformParameterMap.find("ResultImagePixelType");

    const auto resultImagePixelType =
      foundResultImagePixelType == transformParameterMap.end() ? "" : foundResultImagePixelType->second.at(0);

    bool        errorOccurred = false;
    bool        movingImageDimensionFound = false;
    std::string exceptionMessage;

    /** Print a start message. */
    elxout << "Running transformix with parameter file \"" << transformParameterFile << "\".\n" << std::endl;

    elx::ForEachSupportedImageType([&errorOccurred,
                                    &movingImageDimensionFound,
                                    &exceptionMessage,
                                    &movingImageDimension,
                                    &resultImagePixelType,
                                    &outFolder,
                                    &argMap,
                                    &transformParameterMap](const auto elxTypedef) {
      using ElxTypedef = decltype(elxTypedef);

      using MovingImageType = typename ElxTypedef::MovingImageType;
      constexpr auto MovingImageDimension = MovingImageType::ImageDimension;

      try
      {
        if ((errorOccurred || !movingImageDimensionFound) &&
            (std::to_string(MovingImageDimension) == movingImageDimension))
        {
          movingImageDimensionFound = true;
          errorOccurred = false;

          /** Set transformix. */
          elx::DefaultConstruct<itk::TransformixFilter<MovingImageType>> transformixFilter;
          transformixFilter.SetOutputDirectory(outFolder);
          transformixFilter.SetLogFileName("transformix.log");

          const auto inArg = argMap.find("-in");

          if (inArg == argMap.cend())
          {
            // Just a dummy image.
            transformixFilter.SetMovingImage(MovingImageType::New());
          }
          else
          {
            const auto movingImage = itk::ReadImage<MovingImageType>(inArg->second);

            transformixFilter.SetMovingImage(movingImage);
          }
          const auto defArg = argMap.find("-def");

          if (defArg != argMap.cend())
          {
            if (defArg->second == "all")
            {
              transformixFilter.ComputeDeformationFieldOn();
            }
            else
            {
              transformixFilter.SetFixedPointSetFileName(defArg->second);
            }
          }
          {
            const auto parameterObject = elx::ParameterObject::New();
            parameterObject->SetParameterMap(transformParameterMap);

            transformixFilter.SetTransformParameterObject(parameterObject);
          }

          // Run transformix.
          transformixFilter.Update();

          if (inArg != argMap.cend())
          {
            const auto outputImage = transformixFilter.GetOutput();

            if ((outputImage == nullptr) || (outputImage->GetBufferPointer() == nullptr))
            {
              errorOccurred = true;
            }
            else
            {
              const bool doCompression = GetSingleParameterValue(transformParameterMap, "CompressResultImage", false);

              if (resultImagePixelType.empty())
              {
                itk::WriteImage(outputImage, outFolder + "/result.mhd", doCompression);
              }
              else
              {
                itk::WriteCastedImage(*outputImage, outFolder + "/result.mhd", resultImagePixelType, doCompression);
              }
            }
          }
          if (transformixFilter.GetComputeDeformationField())
          {
            const auto * const deformationField = transformixFilter.GetOutputDeformationField();

            if (deformationField != nullptr)
            {
              itk::WriteImage(deformationField, outFolder + "/deformationField.mhd");
            }
          }
        }
      }
      catch (const std::exception & stdException)
      {
        errorOccurred = true;
        exceptionMessage = stdException.what();
        // Just continue with the next iteration of ForEachSupportedImageType.
      }
    });

    if (!movingImageDimensionFound)
    {
      xl::xout["error"] << "Error: unsupported image dimensionality: " << movingImageDimension << std::endl;
    }

    if (errorOccurred)
    {
      xl::xout["error"] << "Error: last exception message:\n" << exceptionMessage << std::endl;
      return EXIT_FAILURE;
    }

    /** Check if transformix run without errors. */
    if (returndummy != 0)
    {
      xl::xout["error"] << "Errors occurred" << std::endl;
      return returndummy;
    }

    /** Stop timer and print it. */
    totaltimer.Stop();
    elxout << "\ntransformix has finished at " << GetCurrentDateAndTime()
           << ".\nTotal time elapsed: " << ConvertSecondsToDHMS(totaltimer.GetMean(), 1) << ".\n"
           << std::endl;

    /** Exit and return the error code. */
    return returndummy;
  }
  catch (const std::exception & stdException)
  {
    elx::ReportTerminatingException("transformix", stdException);
  }
  catch (...)
  {
    assert(!"Exceptions should be derived from std::exception!");
  }
  return EXIT_FAILURE;

} // end main
