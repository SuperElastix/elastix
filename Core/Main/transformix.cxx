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
#include "elxMainExeUtilities.h"
#include "elxTransformixMain.h"
#include <Core/elxVersionMacros.h>
#include "itkUseMevisDicomTiff.h"

// ITK header files:
#include <itkTimeProbe.h>

// Standard C++ header files:
#include <cassert>
#include <iostream>


constexpr const char * transformixHelpText =
  /** Print the version. */
  "transformix version: " ELASTIX_VERSION_STRING "\n\n"

  /** What is transformix? */
  "transformix applies a transform on an input image and/or "
  "generates a deformation field.\n"
  "The transform is specified in the transform-parameter file.\n"
  "  --help, -h displays this message and exit\n"
  "  --version  output version information and exit\n\n"

  /** Mandatory arguments. */
  "Call transformix from the command line with mandatory arguments:\n"
  "  -out      output directory\n"
  "  -tp       transform-parameter file, only 1\n\n"

  /** Optional arguments. */
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
  "  -loglevel set the log level to \"off\", \"error\", \"warning\", or \"info\" (default),\n"
  "  -priority set the process priority to high, abovenormal, normal (default),\n"
  "            belownormal, or idle (Windows only option)\n"
  "  -threads  set the maximum number of threads of transformix\n"
  "\nAt least one of the options \"-in\", \"-def\", \"-jac\", or \"-jacmat\" should be given.\n\n"

  /** The parameter file. */
  "The transform-parameter file must contain all the information "
  "necessary for transformix to run properly. That includes which transform "
  "to use, with which parameters, etc. For a usable transform-parameter file, "
  "run elastix, and inspect the output file \"TransformParameters.0.txt\".\n\n"

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

    /** Check if "-help" or "--version" was asked for.*/
    if (argc == 1)
    {
      std::cout << "Use \"transformix --help\" for information about transformix-usage." << std::endl;
      return 0;
    }
    else if (argc == 2)
    {
      std::string argument(argv[1]);
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

    /** Some typedef's.*/
    using TransformixMainType = elx::TransformixMain;
    using TransformixMainPointer = TransformixMainType::Pointer;
    using ArgumentMapType = TransformixMainType::ArgumentMapType;
    using ArgumentMapEntryType = ArgumentMapType::value_type;

    /** Support Mevis Dicom Tiff (if selected in cmake) */
    RegisterMevisDicomTiff();

    /** Declare an instance of the Transformix class. */
    TransformixMainPointer transformix;

    /** Initialize. */
    int             returndummy = 0;
    ArgumentMapType argMap;
    bool            outFolderPresent = false;
    std::string     outFolder = "";
    std::string     logFileName = "";
    auto            level = elx::log::level::info;

    /** Put command line parameters into parameterFileList. */
    for (unsigned int i = 1; static_cast<std::int64_t>(i) < argc - 1; i += 2)
    {
      std::string key(argv[i]);
      std::string value(argv[i + 1]);

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
          outFolderPresent = true;
          outFolder = value;

        } // end if key == "-out"
      }
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

    } // end for loop

    /** The argv0 argument, required for finding the component.dll/so's. */
    argMap.insert(ArgumentMapEntryType("-argv0", argv[0]));

    /** Check that the option "-tp" is given. */
    if (argMap.count("-tp") == 0)
    {
      std::cerr << "ERROR: No CommandLine option \"-tp\" given!" << std::endl;
      returndummy |= -1;
    }

    /** Check that at least one of the following options is given. */
    if (argMap.count("-in") == 0 && argMap.count("-ipp") == 0 && argMap.count("-def") == 0 &&
        argMap.count("-jac") == 0 && argMap.count("-jacmat") == 0)
    {
      std::cerr
        << "ERROR: At least one of the CommandLine options \"-in\", \"-def\", \"-jac\", or \"-jacmat\" should be given!"
        << std::endl;
      returndummy |= -1;
    }

    /** Check if the -out option is given and setup the log system. */
    if (outFolderPresent)
    {
      /** Check if the output directory exists. */
      bool outFolderExists = itksys::SystemTools::FileIsDirectory(outFolder);
      if (!outFolderExists)
      {
        std::cerr << "ERROR: the output directory \"" << outFolder << "\" does not exist.\n"
                  << "You are responsible for creating it." << std::endl;
        returndummy |= -2;
      }
      else
      {
        /** Setup the log system. */
        logFileName = argMap["-out"] + "transformix.log";
        int returndummy2 = elx::log::setup(logFileName, true, true, level) ? 0 : 1;
        if (returndummy2)
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
    if (returndummy)
    {
      return returndummy;
    }

    elx::log::info("");

    /** Declare a timer, start it and print the start time. */
    itk::TimeProbe totaltimer;
    totaltimer.Start();
    elx::log::info(std::ostringstream{} << "transformix is started at " << GetCurrentDateAndTime() << ".\n");

    // Print where transformix was run, and print its version information.
    elx::log::info(std::ostringstream{} << "which transformix:   " << argv[0] << '\n'
                                        << elx::GetExtendedVersionInformation("transformix", "  ")
                                        << elx::MakeStringOfCommandLineArguments(argv));

    itksys::SystemInformation info;
    info.RunCPUCheck();
    info.RunOSCheck();
    info.RunMemoryCheck();
    elx::log::info(std::ostringstream{} << "transformix runs at: " << info.GetHostname() << '\n'
                                        << "  " << info.GetOSName() << " " << info.GetOSRelease()
                                        << (info.Is64Bits() ? " (x64), " : ", ") << info.GetOSVersion() << '\n'
                                        << "  with " << info.GetTotalPhysicalMemory() << " MB memory, and "
                                        << info.GetNumberOfPhysicalCPU() << " cores @ "
                                        << static_cast<unsigned int>(info.GetProcessorClockFrequency()) << " MHz.");

    /**
     * ********************* START TRANSFORMATION *******************
     */

    /** Set transformix. */
    transformix = TransformixMainType::New();

    /** Print a start message. */
    elx::log::info(std::ostringstream{} << "Running transformix with parameter file \"" << argMap["-tp"] << "\".\n");

    /** Run transformix. */
    returndummy = transformix->Run(argMap);

    /** Check if transformix run without errors. */
    if (returndummy != 0)
    {
      elx::log::error("Errors occurred");
      return returndummy;
    }

    /** Stop timer and print it. */
    totaltimer.Stop();
    elx::log::info(std::ostringstream{} << "\ntransformix has finished at " << GetCurrentDateAndTime() << ".\n"
                                        << "Total time elapsed: " << ConvertSecondsToDHMS(totaltimer.GetMean(), 1)
                                        << ".\n");

    /** Clean up. */
    transformix = nullptr;

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
