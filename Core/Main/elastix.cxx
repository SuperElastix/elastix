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
  " * the elastix website: https://elastix.lumc.nl\n"
  " * the source code repository site: https://github.com/SuperElastix/elastix\n"
  " * the discussion forum: https://groups.google.com/g/elastix-imageregistration";


int
main(int argc, char ** argv)
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
      std::cout << "elastix version: " ELASTIX_VERSION_STRING << "\nITK version: " << ITK_VERSION_MAJOR << '.'
                << ITK_VERSION_MINOR << '.' << ITK_VERSION_PATCH << "\nBuild date: " << __DATE__ << ' ' << __TIME__
#ifdef _MSC_FULL_VER
                << "\nCompiler: Visual C++ version " << _MSC_FULL_VER << '.' << _MSC_BUILD
#endif
#ifdef __clang__
                << "\nCompiler: Clang"
#  ifdef __VERSION__
                << " version " << __VERSION__
#  endif
#endif
#if defined(__GNUC__)
                << "\nCompiler: GCC"
#  ifdef __VERSION__
                << " version " << __VERSION__
#  endif
#endif
                << "\nMemory address size: " << std::numeric_limits<std::size_t>::digits
                << "-bit\nCMake version: " << ELX_CMAKE_VERSION << std::endl;
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
  using FlatDirectionCosinesType = ElastixMainType::FlatDirectionCosinesType;

  using ArgumentMapType = ElastixMainType::ArgumentMapType;
  using ArgumentMapEntryType = ArgumentMapType::value_type;

  /** Support Mevis Dicom Tiff (if selected in cmake) */
  RegisterMevisDicomTiff();

  ArgumentMapType         argMap;
  std::queue<std::string> parameterFileList;
  std::string             outFolder;

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
        std::cerr << "WARNING!" << std::endl;
        std::cerr << "Argument " << key << "is only required once." << std::endl;
        std::cerr << "Arguments " << key << " " << value << "are ignored" << std::endl;
      }

    } // end else (so, if key does not equal "-p")

  } // end for loop

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
      std::cerr << "ERROR: the output directory \"" << outFolder << "\" does not exist." << std::endl;
      std::cerr << "You are responsible for creating it." << std::endl;
      returndummy |= -2;
    }
    else
    {
      /** Setup xout. */
      const std::string logFileName = outFolder + "elastix.log";
      const int         returndummy2{ elx::xoutSetup(logFileName.c_str(), true, true) };
      if (returndummy2 != 0)
      {
        std::cerr << "ERROR while setting up xout." << std::endl;
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

  elxout << std::endl;

  /** Declare a timer, start it and print the start time. */
  itk::TimeProbe totaltimer;
  totaltimer.Start();
  elxout << "elastix is started at " << GetCurrentDateAndTime() << ".\n" << std::endl;

  /** Print where elastix was run. */
  elxout << "which elastix:   " << argv[0] << std::endl;
  itksys::SystemInformation info;
  info.RunCPUCheck();
  info.RunOSCheck();
  info.RunMemoryCheck();
  elxout << "elastix runs at: " << info.GetHostname() << std::endl;
  elxout << "  " << info.GetOSName() << " " << info.GetOSRelease() << (info.Is64Bits() ? " (x64), " : ", ")
         << info.GetOSVersion() << std::endl;
  elxout << "  with " << info.GetTotalPhysicalMemory() << " MB memory, and " << info.GetNumberOfPhysicalCPU()
         << " cores @ " << static_cast<unsigned int>(info.GetProcessorClockFrequency()) << " MHz." << std::endl;


  ObjectPointer              transform = nullptr;
  DataObjectContainerPointer fixedImageContainer = nullptr;
  DataObjectContainerPointer movingImageContainer = nullptr;
  DataObjectContainerPointer fixedMaskContainer = nullptr;
  DataObjectContainerPointer movingMaskContainer = nullptr;
  FlatDirectionCosinesType   fixedImageOriginalDirection;

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
    elastixMain->SetOriginalFixedImageDirectionFlat(fixedImageOriginalDirection);

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
    elxout << "-------------------------------------------------------------------------\n" << std::endl;
    elxout << "Running elastix with parameter file " << i << ": \"" << parameterFileName << "\".\n" << std::endl;

    /** Declare a timer, start it and print the start time. */
    itk::TimeProbe timer;
    timer.Start();
    elxout << "Current time: " << GetCurrentDateAndTime() << "." << std::endl;

    /** Start registration. */
    returndummy = elastixMain->Run(argMap);

    /** Check for errors. */
    if (returndummy != 0)
    {
      xl::xout["error"] << "Errors occurred!" << std::endl;
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
    fixedImageOriginalDirection = elastixMain->GetOriginalFixedImageDirectionFlat();

    /** Print a finish message. */
    elxout << "Running elastix with parameter file " << i << ": \"" << parameterFileName << "\", has finished.\n"
           << std::endl;

    /** Stop timer and print it. */
    timer.Stop();
    elxout << "\nCurrent time: " << GetCurrentDateAndTime() << "." << std::endl;
    elxout << "Time used for running elastix with this parameter file:\n  " << ConvertSecondsToDHMS(timer.GetMean(), 1)
           << ".\n"
           << std::endl;
  } // end loop over registrations

  elxout << "-------------------------------------------------------------------------\n" << std::endl;

  /** Stop totaltimer and print it. */
  totaltimer.Stop();
  elxout << "Total time elapsed: " << ConvertSecondsToDHMS(totaltimer.GetMean(), 1) << ".\n" << std::endl;

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

} // end main
