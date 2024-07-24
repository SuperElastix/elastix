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
#include "elastix.h"
#include "elxConversion.h"
#include "elxElastixMain.h"
#include "elxMainExeUtilities.h"
#include <Core/elxVersionMacros.h>
#include "itkUseMevisDicomTiff.h"

// ITK header files:
#include <itkTimeProbe.h>
#include <itksys/SystemInformation.hxx>
#include <itksys/SystemTools.hxx>

// Standard C++ header files:
#include <cassert>
#include <climits> // For UINT_MAX.
#include <cstddef> // For size_t.
#include <iostream>
#include <limits>
#include <queue>
#include <vector>


constexpr const char * elastixHelpText =
  /** Print the version. */
  "elastix version: " ELASTIX_VERSION_STRING "\n\n"

  /** What is elastix? */
  "elastix registers a moving image to a fixed image.\n"
  "The registration-process is specified in the parameter file.\n"
  "  --help, -h displays this message and exit\n"
  "  --version  output version information and exit\n"
  "  --extended-version  output extended version information and exit\n\n"

  /** Mandatory arguments.*/
  "Call elastix from the command line with mandatory arguments:\n"
  "  -f        fixed image\n"
  "  -m        moving image\n"
  "  -out      output directory\n"
  "  -p        parameter file, elastix handles 1 or more \"-p\"\n\n"

  /** Optional arguments.*/
  "Optional extra commands:\n"
  "  -fMask    mask for fixed image\n"
  "  -mMask    mask for moving image\n"
  "  -fp       point set for fixed image\n"
  "  -mp       point set for moving image\n"
  "  -t0       parameter file for initial transform\n"
  "  -loglevel set the log level to \"off\", \"error\", \"warning\", or \"info\" (default),\n"
  "  -priority set the process priority to high, abovenormal, normal (default),\n"
  "            belownormal, or idle (Windows only option)\n"
  "  -threads  set the maximum number of threads of elastix\n\n"

  /** The parameter file.*/
  "The parameter-file must contain all the information "
  "necessary for elastix to run properly. That includes which metric to "
  "use, which optimizer, which transform, etc. It must also contain "
  "information specific for the metric, optimizer, transform, etc. "
  "For a usable parameter-file, see the website.\n\n"

  "Need further help? Please check:\n"
  " * the elastix website: https://elastix.dev\n"
  " * the source code repository site: https://github.com/SuperElastix/elastix\n"
  " * the discussion forum: https://groups.google.com/g/elastix-imageregistration";


int
main(int argc, char ** argv)
{
  try
  {
    elastix::BaseComponent::InitializeElastixExecutable();
    assert(!elastix::BaseComponent::IsElastixLibrary());

    /** Check if "--help" or "--version" was asked for. */
    if (argc == 1)
    {
      std::cout << "Use \"elastix --help\" for information about elastix-usage." << std::endl;
      return 0;
    }
    else if (argc == 2)
    {
      std::string argument(argv[1]);
      if (argument == "-help" || argument == "--help" || argument == "-h")
      {
        std::cout << elastixHelpText << std::endl;
        return 0;
      }
      else if (argument == "--version")
      {
        std::cout << "elastix version: " ELASTIX_VERSION_STRING << std::endl;
        return 0;
      }
      else if (argument == "--extended-version")
      {
        std::cout << elx::GetExtendedVersionInformation("elastix");
        return 0;
      }
      else
      {
        std::cout << "Use \"elastix --help\" for information about elastix-usage." << std::endl;
        return 0;
      }
    }

    /** Some typedef's. */
    using ElastixMainType = elx::ElastixMain;
    using ObjectPointer = ElastixMainType::ObjectPointer;
    using DataObjectContainerPointer = ElastixMainType::DataObjectContainerPointer;

    using ArgumentMapType = ElastixMainType::ArgumentMapType;
    using ArgumentMapEntryType = ArgumentMapType::value_type;

    /** Support Mevis Dicom Tiff (if selected in cmake) */
    RegisterMevisDicomTiff();

    ArgumentMapType         argMap;
    std::queue<std::string> parameterFileList;
    std::string             outFolder;
    auto                    level = elx::log::level::info;

    /** Put command line parameters into parameterFileList. */
    for (unsigned int i = 1; static_cast<long>(i) < (argc - 1); i += 2)
    {
      std::string key(argv[i]);
      std::string value(argv[i + 1]);

      if (key == "-p")
      {
        /** Queue the ParameterFileNames. */
        parameterFileList.push(value);
        /** The different '-p' are stored in the argMap, with
         * keys p(1), p(2), etc. */
        std::ostringstream tempPname;
        tempPname << "-p(" << parameterFileList.size() << ")";
        std::string tempPName = tempPname.str();
        argMap.insert(ArgumentMapEntryType(tempPName, value));
      }
      else
      {
        if (key == "-loglevel")
        {
          if (!ToLogLevel(value, level))
          {
            // Unsupported log level value.
            return EXIT_FAILURE;
          }
        }
        else
        {
          if (key == "-out")
          {
            /** Make sure that last character of the output folder equals a '/' or '\'. */
            const char last = value.back();
            if (last != '/' && last != '\\')
            {
              value.append("/");
            }
            value = elx::Conversion::ToNativePathNameSeparators(value);

            /** Save this information. */
            outFolder = value;

          } // end if key == "-out"

          /** Attempt to save the arguments in the ArgumentMap. */
          if (argMap.count(key) == 0)
          {
            argMap.insert(ArgumentMapEntryType(key, value));
          }
          else
          {
            /** Duplicate arguments. */
            std::cerr << "WARNING!\n"
                      << "Argument " << key << "is only required once.\n"
                      << "Arguments " << key << " " << value << "are ignored" << std::endl;
          }
        }
      } // end else (so, if key does not equal "-p")
    }   // end for loop

    /** The argv0 argument, required for finding the component.dll/so's. */
    argMap.insert(ArgumentMapEntryType("-argv0", argv[0]));

    int returndummy{};

    /** Check if at least once the option "-p" is given. */
    if (parameterFileList.empty())
    {
      std::cerr << "ERROR: No CommandLine option \"-p\" given!" << std::endl;
      returndummy |= -1;
    }

    /** Check if the -out option is given. */
    if (!outFolder.empty())
    {
      /** Check if the output directory exists. */
      if (!itksys::SystemTools::FileIsDirectory(outFolder))
      {
        std::cerr << "ERROR: the output directory \"" << outFolder << "\" does not exist.\n"
                  << "You are responsible for creating it." << std::endl;
        returndummy |= -2;
      }
      else
      {
        /** Setup the log system. */
        const std::string logFileName = outFolder + "elastix.log";
        const int         returndummy2 = elx::log::setup(logFileName, true, true, level) ? 0 : 1;

        if (returndummy2 != 0)
        {
          std::cerr << "ERROR while setting up the log system." << std::endl;
        }
        returndummy |= returndummy2;
      }
    }
    else
    {
      returndummy = -2;
      std::cerr << "ERROR: No CommandLine option \"-out\" given!" << std::endl;
    }

    /** Stop if some fatal errors occurred. */
    if (returndummy != 0)
    {
      return returndummy;
    }

    elx::log::info("");

    /** Declare a timer, start it and print the start time. */
    itk::TimeProbe totaltimer;
    totaltimer.Start();
    elx::log::info(std::ostringstream{} << "elastix is started at " << GetCurrentDateAndTime() << ".\n");

    // Print where elastix was run, and print its version information.
    elx::log::info(std::ostringstream{} << "which elastix:   " << argv[0] << '\n'
                                        << elx::GetExtendedVersionInformation("elastix", "  ")
                                        << elx::MakeStringOfCommandLineArguments(argv));

    itksys::SystemInformation info;
    info.RunCPUCheck();
    info.RunOSCheck();
    info.RunMemoryCheck();
    elx::log::info(std::ostringstream{} << "elastix runs at: " << info.GetHostname() << '\n'
                                        << "  " << info.GetOSName() << " " << info.GetOSRelease()
                                        << (info.Is64Bits() ? " (x64), " : ", ") << info.GetOSVersion() << '\n'
                                        << "  with " << info.GetTotalPhysicalMemory() << " MB memory, and "
                                        << info.GetNumberOfPhysicalCPU() << " cores @ "
                                        << static_cast<unsigned int>(info.GetProcessorClockFrequency()) << " MHz.");


    ObjectPointer                             transform = nullptr;
    DataObjectContainerPointer                fixedImageContainer = nullptr;
    DataObjectContainerPointer                movingImageContainer = nullptr;
    DataObjectContainerPointer                fixedMaskContainer = nullptr;
    DataObjectContainerPointer                movingMaskContainer = nullptr;
    ElastixMainType::FlatDirectionCosinesType fixedImageOriginalDirectionFlat;

    /**
     * ********************* START REGISTRATION *********************
     *
     * Do the (possibly multiple) registration(s).
     */

    const auto nrOfParameterFiles = parameterFileList.size();
    assert(nrOfParameterFiles <= UINT_MAX);

    for (unsigned i{}; i < static_cast<unsigned>(nrOfParameterFiles); ++i)
    {
      /** Create another instance of ElastixMain. */
      const auto elastixMain = ElastixMainType::New();

      /** Set stuff we get from a former registration. */
      elastixMain->SetInitialTransform(transform);
      elastixMain->SetFixedImageContainer(fixedImageContainer);
      elastixMain->SetMovingImageContainer(movingImageContainer);
      elastixMain->SetFixedMaskContainer(fixedMaskContainer);
      elastixMain->SetMovingMaskContainer(movingMaskContainer);
      elastixMain->SetOriginalFixedImageDirectionFlat(fixedImageOriginalDirectionFlat);

      /** Set the current elastix-level. */
      elastixMain->SetElastixLevel(i);
      elastixMain->SetTotalNumberOfElastixLevels(nrOfParameterFiles);

      /** Get the argMap entry for the parameter file, and exchange its file name
       * with the first file name in the list.
       */
      std::string & parameterFileName = argMap["-p"];
      parameterFileName.swap(parameterFileList.front());
      parameterFileList.pop();

      /** Print a start message. */
      elx::log::info(std::ostringstream{}
                     << "-------------------------------------------------------------------------\n"
                     << '\n'
                     << "Running elastix with parameter file " << i << ": \"" << parameterFileName << "\".\n");

      /** Declare a timer, start it and print the start time. */
      itk::TimeProbe timer;
      timer.Start();
      elx::log::info(std::ostringstream{} << "Current time: " << GetCurrentDateAndTime() << ".");

      /** Start registration. */
      returndummy = elastixMain->Run(argMap);

      /** Check for errors. */
      if (returndummy != 0)
      {
        elx::log::error("Errors occurred!");
        return returndummy;
      }

      /** Get the transform, the fixedImage and the movingImage
       * in order to put it in the (possibly) next registration.
       */
      transform = elastixMain->GetModifiableFinalTransform();
      fixedImageContainer = elastixMain->GetModifiableFixedImageContainer();
      movingImageContainer = elastixMain->GetModifiableMovingImageContainer();
      fixedMaskContainer = elastixMain->GetModifiableFixedMaskContainer();
      movingMaskContainer = elastixMain->GetModifiableMovingMaskContainer();
      fixedImageOriginalDirectionFlat = elastixMain->GetOriginalFixedImageDirectionFlat();

      /** Print a finish message. */
      elx::log::info(std::ostringstream{} << "Running elastix with parameter file " << i << ": \"" << parameterFileName
                                          << "\", has finished.\n");

      /** Stop timer and print it. */
      timer.Stop();
      elx::log::info(std::ostringstream{} << "\nCurrent time: " << GetCurrentDateAndTime() << ".\n"
                                          << "Time used for running elastix with this parameter file:\n  "
                                          << ConvertSecondsToDHMS(timer.GetMean(), 1) << ".\n");
    } // end loop over registrations

    elx::log::info("-------------------------------------------------------------------------\n");

    /** Stop totaltimer and print it. */
    totaltimer.Stop();
    elx::log::info(std::ostringstream{} << "Total time elapsed: " << ConvertSecondsToDHMS(totaltimer.GetMean(), 1)
                                        << ".\n");

    /**
     * Make sure all the components that are defined in a Module (.DLL/.so)
     * are deleted before the modules are closed.
     */

    transform = nullptr;
    fixedImageContainer = nullptr;
    movingImageContainer = nullptr;
    fixedMaskContainer = nullptr;
    movingMaskContainer = nullptr;

    /** Exit and return the error code. */
    return 0;
  }
  catch (const std::exception & stdException)
  {
    elx::ReportTerminatingException("elastix", stdException);
  }
  catch (...)
  {
    assert(!"Exceptions should be derived from std::exception!");
  }
  return EXIT_FAILURE;

} // end main
