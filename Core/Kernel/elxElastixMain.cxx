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

/** If running on a Windows-system, include "windows.h".
 *  This is to set the priority, but which does not work on cygwin.
 */

#if defined(_WIN32) && !defined(__CYGWIN__)
#  include <windows.h>
#endif

#include "elxElastixMain.h"
#include "elxComponentLoader.h"

#include "elxMacro.h"
#include "itkPlatformMultiThreader.h"

#ifdef ELASTIX_USE_OPENCL
#  include "itkOpenCLContext.h"
#  include "itkOpenCLSetup.h"
#endif

namespace
{
/**
 * ******************* Global variables *************************
 *
 * Some global variables (not part of the ElastixMain class, used
 * by xoutSetup.
 */

struct Data
{
  /** xout TargetCells. */
  xl::xoutmain   Xout;
  xl::xoutsimple WarningXout;
  xl::xoutsimple ErrorXout;
  xl::xoutsimple StandardXout;
  xl::xoutsimple CoutOnlyXout;
  xl::xoutsimple LogOnlyXout;
  std::ofstream  LogFileStream;
};

Data g_data;

} // end unnamed namespace

/**
 * ********************* xoutSetup ******************************
 *
 * NB: this function is a global function, not part of the ElastixMain
 * class!!
 */

int
elastix::xoutSetup(const char * logfilename, bool setupLogging, bool setupCout)
{
  int returndummy = 0;
  set_xout(&g_data.Xout);

  if (setupLogging)
  {
    /** Open the logfile for writing. */
    g_data.LogFileStream.open(logfilename);
    if (!g_data.LogFileStream.is_open())
    {
      std::cerr << "ERROR: LogFile cannot be opened!" << std::endl;
      return 1;
    }
  }

  /** Set std::cout and the logfile as outputs of xout. */
  if (setupLogging)
  {
    returndummy |= xl::xout.AddOutput("log", &g_data.LogFileStream);
  }
  if (setupCout)
  {
    returndummy |= xl::xout.AddOutput("cout", &std::cout);
  }

  /** Set outputs of LogOnly and CoutOnly. */
  returndummy |= g_data.LogOnlyXout.AddOutput("log", &g_data.LogFileStream);
  returndummy |= g_data.CoutOnlyXout.AddOutput("cout", &std::cout);

  /** Copy the outputs to the warning-, error- and standard-xouts. */
  g_data.WarningXout.SetOutputs(xl::xout.GetCOutputs());
  g_data.ErrorXout.SetOutputs(xl::xout.GetCOutputs());
  g_data.StandardXout.SetOutputs(xl::xout.GetCOutputs());

  g_data.WarningXout.SetOutputs(xl::xout.GetXOutputs());
  g_data.ErrorXout.SetOutputs(xl::xout.GetXOutputs());
  g_data.StandardXout.SetOutputs(xl::xout.GetXOutputs());

  /** Link the warning-, error- and standard-xouts to xout. */
  returndummy |= xl::xout.AddTargetCell("warning", &g_data.WarningXout);
  returndummy |= xl::xout.AddTargetCell("error", &g_data.ErrorXout);
  returndummy |= xl::xout.AddTargetCell("standard", &g_data.StandardXout);
  returndummy |= xl::xout.AddTargetCell("logonly", &g_data.LogOnlyXout);
  returndummy |= xl::xout.AddTargetCell("coutonly", &g_data.CoutOnlyXout);

  /** Format the output. */
  xl::xout["standard"] << std::fixed;
  xl::xout["standard"] << std::showpoint;

  /** Return a value. */
  return returndummy;

} // end xoutSetup()


namespace elastix
{

/**
 * ********************* xoutManager ******************************
 */

xoutManager::xoutManager(const std::string & logFileName, const bool setupLogging, const bool setupCout)
{
  if (xoutSetup(logFileName.c_str(), setupLogging, setupCout))
  {
    itkGenericExceptionMacro("Error while setting up xout");
  }
}

xoutManager::Guard::~Guard()
{
  g_data = {};
}


/**
 * ********************* Constructor ****************************
 */

ElastixMain::ElastixMain()
{
  /** Initialize the components. */
  this->m_Configuration = ConfigurationType::New();

  this->m_Elastix = nullptr;

  this->m_FixedImagePixelType = "";
  this->m_FixedImageDimension = 0;

  this->m_MovingImagePixelType = "";
  this->m_MovingImageDimension = 0;

  this->m_DBIndex = 0;

  this->m_FixedImageContainer = nullptr;
  this->m_MovingImageContainer = nullptr;

  this->m_FixedMaskContainer = nullptr;
  this->m_MovingMaskContainer = nullptr;

  this->m_ResultImageContainer = nullptr;

  this->m_FinalTransform = nullptr;
  this->m_InitialTransform = nullptr;
  this->m_TransformParametersMap.clear();

} // end Constructor


/**
 * ****************** GetComponentDatabase *********
 */

const ComponentDatabase &
ElastixMain::GetComponentDatabase(void)
{
  // Improved thread-safety by using C++11 "magic statics".
  static const auto componentDatabase = [] {
    const auto componentDatabase = ComponentDatabase::New();
    const auto componentLoader = ComponentLoader::New();
    componentLoader->SetComponentDatabase(componentDatabase);

    if (componentLoader->LoadComponents() != 0)
    {
      xl::xout["error"] << "Loading components failed" << std::endl;
    }
    return componentDatabase;
  }();
  return *componentDatabase;
}


/**
 * ********************** Destructor ****************************
 */

ElastixMain::~ElastixMain()
{
#ifdef ELASTIX_USE_OPENCL
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  if (context->IsCreated())
  {
    context->Release();
  }
#endif
} // end Destructor


/**
 * *************** EnterCommandLineParameters *******************
 */

void
ElastixMain::EnterCommandLineArguments(const ArgumentMapType & argmap)
{

  /** Initialize the configuration object with the
   * command line parameters entered by the user.
   */
  int dummy = this->m_Configuration->Initialize(argmap);
  if (dummy)
  {
    xl::xout["error"] << "ERROR: Something went wrong during initialization "
                      << "of the configuration object." << std::endl;
  }

} // end EnterCommandLineParameters()


/**
 * *************** EnterCommandLineArguments *******************
 */

void
ElastixMain::EnterCommandLineArguments(const ArgumentMapType & argmap, const ParameterMapType & inputMap)
{
  /** Initialize the configuration object with the
   * command line parameters entered by the user.
   */
  int dummy = this->m_Configuration->Initialize(argmap, inputMap);
  if (dummy)
  {
    xl::xout["error"] << "ERROR: Something went wrong during initialization of the configuration object." << std::endl;
  }

} // end EnterCommandLineArguments()


/**
 * *************** EnterCommandLineArguments *******************
 */

void
ElastixMain::EnterCommandLineArguments(const ArgumentMapType & argmap, const std::vector<ParameterMapType> & inputMaps)
{
  this->m_Configurations.clear();
  this->m_Configurations.resize(inputMaps.size());

  for (size_t i = 0; i < inputMaps.size(); ++i)
  {
    /** Initialize the configuration object with the
     * command line parameters entered by the user.
     */
    this->m_Configurations[i] = ConfigurationType::New();
    int dummy = this->m_Configurations[i]->Initialize(argmap, inputMaps[i]);
    if (dummy)
    {
      xl::xout["error"] << "ERROR: Something went wrong during initialization of configuration object " << i << "."
                        << std::endl;
    }
  }

  /** Copy last configuration object to m_Configuration. */
  this->m_Configuration = this->m_Configurations[inputMaps.size() - 1];
} // end EnterCommandLineArguments()


/**
 * **************************** Run *****************************
 *
 * Assuming EnterCommandLineParameters has already been invoked.
 * or that m_Configuration is initialized in another way.
 */

int
ElastixMain::Run(void)
{

  /** Set process properties. */
  this->SetProcessPriority();
  this->SetMaximumNumberOfThreads();

  /** Initialize database. */
  int errorCode = this->InitDBIndex();
  if (errorCode != 0)
  {
    return errorCode;
  }

  /** Create the elastix component. */
  try
  {
    /** Key "Elastix", see elxComponentLoader::InstallSupportedImageTypes(). */
    this->m_Elastix = this->CreateComponent("Elastix");
  }
  catch (itk::ExceptionObject & excp)
  {
    /** We just print the exception and let the program quit. */
    xl::xout["error"] << excp << std::endl;
    errorCode = 1;
    return errorCode;
  }

  /** Create OpenCL context and logger here. */
#ifdef ELASTIX_USE_OPENCL
  /** Check if user overrides OpenCL device selection. */
  std::string userSuppliedOpenCLDeviceType = "GPU";
  this->m_Configuration->ReadParameter(userSuppliedOpenCLDeviceType, "OpenCLDeviceType", 0, false);

  int userSuppliedOpenCLDeviceID = -1;
  this->m_Configuration->ReadParameter(userSuppliedOpenCLDeviceID, "OpenCLDeviceID", 0, false);

  std::string errorMessage = "";
  const bool  creatingContextSuccessful =
    itk::CreateOpenCLContext(errorMessage, userSuppliedOpenCLDeviceType, userSuppliedOpenCLDeviceID);
  if (!creatingContextSuccessful)
  {
    /** Report and disable the GPU by releasing the context. */
    elxout << errorMessage << std::endl;
    elxout << "  OpenCL processing in elastix is disabled." << std::endl << std::endl;

    itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
    context->Release();
  }

  /** Create a log file. */
  itk::CreateOpenCLLogger("elastix", this->m_Configuration->GetCommandLineArgument("-out"));
#endif
  auto & elastixBase = this->GetElastixBase();

  /** Set some information in the ElastixBase. */
  elastixBase.SetConfiguration(this->m_Configuration);
  elastixBase.SetDBIndex(this->m_DBIndex);

  /** Populate the component containers. ImageSampler is not mandatory.
   * No defaults are specified for ImageSampler, Metric, Transform
   * and Optimizer.
   */
  elastixBase.SetRegistrationContainer(
    this->CreateComponents("Registration", "MultiResolutionRegistration", errorCode));

  elastixBase.SetFixedImagePyramidContainer(
    this->CreateComponents("FixedImagePyramid", "FixedSmoothingImagePyramid", errorCode));

  elastixBase.SetMovingImagePyramidContainer(
    this->CreateComponents("MovingImagePyramid", "MovingSmoothingImagePyramid", errorCode));

  elastixBase.SetImageSamplerContainer(this->CreateComponents("ImageSampler", "", errorCode, false));

  elastixBase.SetInterpolatorContainer(this->CreateComponents("Interpolator", "BSplineInterpolator", errorCode));

  elastixBase.SetMetricContainer(this->CreateComponents("Metric", "", errorCode));

  elastixBase.SetOptimizerContainer(this->CreateComponents("Optimizer", "", errorCode));

  elastixBase.SetResampleInterpolatorContainer(
    this->CreateComponents("ResampleInterpolator", "FinalBSplineInterpolator", errorCode));

  elastixBase.SetResamplerContainer(this->CreateComponents("Resampler", "DefaultResampler", errorCode));

  elastixBase.SetTransformContainer(this->CreateComponents("Transform", "", errorCode));

  /** Check if all component could be created. */
  if (errorCode != 0)
  {
    xl::xout["error"] << "ERROR:" << std::endl;
    xl::xout["error"] << "One or more components could not be created." << std::endl;
    return 1;
  }

  /** Set the images and masks. If not set by the user, it is not a problem.
   * ElastixTemplate will try to load them from disk.
   */
  elastixBase.SetFixedImageContainer(this->GetModifiableFixedImageContainer());
  elastixBase.SetMovingImageContainer(this->GetModifiableMovingImageContainer());
  elastixBase.SetFixedMaskContainer(this->GetModifiableFixedMaskContainer());
  elastixBase.SetMovingMaskContainer(this->GetModifiableMovingMaskContainer());
  elastixBase.SetResultImageContainer(this->GetModifiableResultImageContainer());

  /** Set the initial transform, if it happens to be there. */
  elastixBase.SetInitialTransform(this->GetModifiableInitialTransform());

  /** Set the original fixed image direction cosines (relevant in case the
   * UseDirectionCosines parameter was set to false.
   */
  elastixBase.SetOriginalFixedImageDirectionFlat(this->GetOriginalFixedImageDirectionFlat());

  /** Run elastix! */
  try
  {
    errorCode = elastixBase.Run();
  }
  catch (itk::ExceptionObject & excp1)
  {
    /** We just print the itk::exception and let the program quit. */
    xl::xout["error"] << excp1 << std::endl;
    errorCode = 1;
  }
  catch (std::exception & excp2)
  {
    /** We just print the std::exception and let the program quit. */
    xl::xout["error"] << "std: " << excp2.what() << std::endl;
    errorCode = 1;
  }
  catch (...)
  {
    /** We don't know what happened and just print a general message. */
    xl::xout["error"] << "ERROR: an unknown non-ITK, non-std exception was caught.\n"
                      << "Please report this to elastix@bigr.nl." << std::endl;
    errorCode = 1;
  }

  /** Return the final transform. */
  this->m_FinalTransform = elastixBase.GetFinalTransform();

  /** Get the transformation parameter map */
  this->m_TransformParametersMap = elastixBase.GetTransformParametersMap();

  /** Store the images in ElastixMain. */
  this->SetFixedImageContainer(elastixBase.GetFixedImageContainer());
  this->SetMovingImageContainer(elastixBase.GetMovingImageContainer());
  this->SetFixedMaskContainer(elastixBase.GetFixedMaskContainer());
  this->SetMovingMaskContainer(elastixBase.GetMovingMaskContainer());
  this->SetResultImageContainer(elastixBase.GetResultImageContainer());

  /** Store the original fixed image direction cosines (relevant in case the
   * UseDirectionCosines parameter was set to false. */
  this->SetOriginalFixedImageDirectionFlat(elastixBase.GetOriginalFixedImageDirectionFlat());

  /** Return a value. */
  return errorCode;

} // end Run()


/**
 * **************************** Run *****************************
 */

int
ElastixMain::Run(const ArgumentMapType & argmap)
{
  this->EnterCommandLineArguments(argmap);
  return this->Run();
} // end Run()


/**
 * **************************** Run *****************************
 */

int
ElastixMain::Run(const ArgumentMapType & argmap, const ParameterMapType & inputMap)
{
  this->EnterCommandLineArguments(argmap, inputMap);
  return this->Run();
} // end Run()


/**
 * ************************** InitDBIndex ***********************
 *
 * Checks if the configuration object has been initialized,
 * determines the requested ImageTypes, and sets the m_DBIndex
 * to the corresponding value (by asking the elx::ComponentDatabase).
 */

int
ElastixMain::InitDBIndex(void)
{
  /** Only do something when the configuration object wasn't initialized yet. */
  if (this->m_Configuration->IsInitialized())
  {
    /** FixedImagePixelType. */
    if (this->m_FixedImagePixelType.empty())
    {
      /** Try to read it from the parameter file. */
      this->m_FixedImagePixelType = "float"; // \note: this assumes elastix was compiled for float
      this->m_Configuration->ReadParameter(this->m_FixedImagePixelType, "FixedInternalImagePixelType", 0);
    }

    /** FixedImageDimension. */
    if (this->m_FixedImageDimension == 0)
    {
      if (!BaseComponent::IsElastixLibrary())
      {
        /** Get the fixed image file name. */
        std::string fixedImageFileName = this->m_Configuration->GetCommandLineArgument("-f");
        if (fixedImageFileName == "")
        {
          fixedImageFileName = this->m_Configuration->GetCommandLineArgument("-f0");
        }

        /** Sanity check. */
        if (fixedImageFileName == "")
        {
          xl::xout["error"] << "ERROR: could not read fixed image." << std::endl;
          xl::xout["error"] << "  both -f and -f0 are unspecified" << std::endl;
          return 1;
        }

        /** Read it from the fixed image header. */
        try
        {
          this->GetImageInformationFromFile(fixedImageFileName, this->m_FixedImageDimension);
        }
        catch (itk::ExceptionObject & err)
        {
          xl::xout["error"] << "ERROR: could not read fixed image." << std::endl;
          xl::xout["error"] << err << std::endl;
          return 1;
        }

        /** Try to read it from the parameter file.
         * This only serves as a check; elastix versions prior to 4.6 read the dimension
         * from the parameter file, but now we read it from the image header.
         */
        unsigned int fixDimParameterFile = 0;
        bool         foundInParameterFile =
          this->m_Configuration->ReadParameter(fixDimParameterFile, "FixedImageDimension", 0, false);

        /** Check. */
        if (foundInParameterFile)
        {
          if (fixDimParameterFile != this->m_FixedImageDimension)
          {
            xl::xout["error"] << "ERROR: problem defining fixed image dimension.\n"
                              << "  The parameter file says:     " << fixDimParameterFile << "\n"
                              << "  The fixed image header says: " << this->m_FixedImageDimension << "\n"
                              << "  Note that from elastix 4.6 the parameter file definition \"FixedImageDimension\" "
                              << "is not needed anymore.\n  Please remove this entry from your parameter file."
                              << std::endl;
            return 1;
          }
        }
      }
      else
      {
        this->m_Configuration->ReadParameter(this->m_FixedImageDimension, "FixedImageDimension", 0, false);
      }

      /** Just a sanity check, probably not needed. */
      if (this->m_FixedImageDimension == 0)
      {
        xl::xout["error"] << "ERROR:" << std::endl;
        xl::xout["error"] << "The FixedImageDimension is not given." << std::endl;
        return 1;
      }
    }

    /** MovingImagePixelType. */
    if (this->m_MovingImagePixelType.empty())
    {
      /** Try to read it from the parameter file. */
      this->m_MovingImagePixelType = "float"; // \note: this assumes elastix was compiled for float
      this->m_Configuration->ReadParameter(this->m_MovingImagePixelType, "MovingInternalImagePixelType", 0);
    }

    /** MovingImageDimension. */
    if (this->m_MovingImageDimension == 0)
    {
      if (!BaseComponent::IsElastixLibrary())
      {
        /** Get the moving image file name. */
        std::string movingImageFileName = this->m_Configuration->GetCommandLineArgument("-m");
        if (movingImageFileName == "")
        {
          movingImageFileName = this->m_Configuration->GetCommandLineArgument("-m0");
        }

        /** Sanity check. */
        if (movingImageFileName == "")
        {
          xl::xout["error"] << "ERROR: could not read moving image." << std::endl;
          xl::xout["error"] << "  both -m and -m0 are unspecified" << std::endl;
          return 1;
        }

        /** Read it from the moving image header. */
        try
        {
          this->GetImageInformationFromFile(movingImageFileName, this->m_MovingImageDimension);
        }
        catch (itk::ExceptionObject & err)
        {
          xl::xout["error"] << "ERROR: could not read moving image." << std::endl;
          xl::xout["error"] << err << std::endl;
          return 1;
        }

        /** Try to read it from the parameter file.
         * This only serves as a check; elastix versions prior to 4.6 read the dimension
         * from the parameter file, but now we read it from the image header.
         */
        unsigned int movDimParameterFile = 0;
        bool         foundInParameterFile =
          this->m_Configuration->ReadParameter(movDimParameterFile, "MovingImageDimension", 0, false);

        /** Check. */
        if (foundInParameterFile)
        {
          if (movDimParameterFile != this->m_MovingImageDimension)
          {
            xl::xout["error"] << "ERROR: problem defining moving image dimension.\n"
                              << "  The parameter file says:      " << movDimParameterFile << "\n"
                              << "  The moving image header says: " << this->m_MovingImageDimension << "\n"
                              << "  Note that from elastix 4.6 the parameter file definition \"MovingImageDimension\" "
                              << "is not needed anymore.\n  Please remove this entry from your parameter file."
                              << std::endl;
            return 1;
          }
        }
      }
      else
      {
        this->m_Configuration->ReadParameter(this->m_MovingImageDimension, "MovingImageDimension", 0, false);
      }

      /** Just a sanity check, probably not needed. */
      if (this->m_MovingImageDimension == 0)
      {
        xl::xout["error"] << "ERROR:" << std::endl;
        xl::xout["error"] << "The MovingImageDimension is not given." << std::endl;
        return 1;
      }
    }

    /** Get the DBIndex from the ComponentDatabase. */
    this->m_DBIndex = GetComponentDatabase().GetIndex(this->m_FixedImagePixelType,
                                                      this->m_FixedImageDimension,
                                                      this->m_MovingImagePixelType,
                                                      this->m_MovingImageDimension);
    if (this->m_DBIndex == 0)
    {
      xl::xout["error"] << "ERROR:" << std::endl;
      xl::xout["error"] << "Something went wrong in the ComponentDatabase" << std::endl;
      return 1;
    }

  } // end if m_Configuration->Initialized();
  else
  {
    xl::xout["error"] << "ERROR:" << std::endl;
    xl::xout["error"] << "The configuration object has not been initialized." << std::endl;
    return 1;
  }

  /** Return an OK value. */
  return 0;

} // end InitDBIndex()


/**
 * ********************* SetElastixLevel ************************
 */

void
ElastixMain::SetElastixLevel(unsigned int level)
{
  /** Call SetElastixLevel from MyConfiguration. */
  this->m_Configuration->SetElastixLevel(level);

} // end SetElastixLevel()


/**
 * ********************* GetElastixLevel ************************
 */

unsigned int
ElastixMain::GetElastixLevel(void)
{
  /** Call GetElastixLevel from MyConfiguration. */
  return this->m_Configuration->GetElastixLevel();

} // end GetElastixLevel()


/**
 * ********************* SetTotalNumberOfElastixLevels ************************
 */

void
ElastixMain::SetTotalNumberOfElastixLevels(unsigned int levels)
{
  /** Call SetTotalNumberOfElastixLevels from MyConfiguration. */
  this->m_Configuration->SetTotalNumberOfElastixLevels(levels);

} // end SetTotalNumberOfElastixLevels()


/**
 * ********************* GetTotalNumberOfElastixLevels ************************
 */

unsigned int
ElastixMain::GetTotalNumberOfElastixLevels(void)
{
  /** Call GetTotalNumberOfElastixLevels from MyConfiguration. */
  return this->m_Configuration->GetTotalNumberOfElastixLevels();

} // end GetTotalNumberOfElastixLevels()


/**
 * ************************* GetElastixBase ***************************
 */

ElastixMain::ElastixBaseType &
ElastixMain::GetElastixBase(void) const
{
  /** Convert ElastixAsObject to a pointer to an ElastixBaseType. */
  const auto testpointer = dynamic_cast<ElastixBaseType *>(this->m_Elastix.GetPointer());
  if (testpointer == nullptr)
  {
    itkExceptionMacro(<< "Probably GetElastixBase() is called before having called Run()");
  }

  return *testpointer;

} // end GetElastixBase()


/**
 * ************************* CreateComponent ***************************
 */

ElastixMain::ObjectPointer
ElastixMain::CreateComponent(const ComponentDescriptionType & name)
{
  /** A pointer to the New() function. */
  PtrToCreator testcreator = nullptr;
  testcreator = GetComponentDatabase().GetCreator(name, this->m_DBIndex);

  // Note that ObjectPointer() yields a default-constructed SmartPointer (null).
  ObjectPointer testpointer = testcreator ? testcreator() : ObjectPointer();

  if (testpointer.IsNull())
  {
    itkExceptionMacro(<< "The following component could not be created: " << name);
  }

  return testpointer;

} // end CreateComponent()


/**
 * *********************** CreateComponents *****************************
 */

ElastixMain::ObjectContainerPointer
ElastixMain::CreateComponents(const std::string &              key,
                              const ComponentDescriptionType & defaultComponentName,
                              int &                            errorcode,
                              bool                             mandatoryComponent)
{
  ComponentDescriptionType componentName = defaultComponentName;
  unsigned int             componentnr = 0;
  ObjectContainerPointer   objectContainer = ObjectContainerType::New();
  objectContainer->Initialize();

  /** Read the component name.
   * If the user hasn't specified any component names, use
   * the default, and give a warning.
   */
  bool found = this->m_Configuration->ReadParameter(componentName, key, componentnr, true);

  /** If the default equals "" (so no default), the mandatoryComponent
   * flag is true, and not component was given by the user,
   * then elastix quits.
   */
  if (!found && (defaultComponentName == ""))
  {
    if (mandatoryComponent)
    {
      xl::xout["error"] << "ERROR: the following component has not been specified: " << key << std::endl;
      errorcode = 1;
      return objectContainer;
    }
    else
    {
      /* Just return an empty container without nagging. */
      errorcode = 0;
      return objectContainer;
    }
  }

  /** Try creating the specified component. */
  try
  {
    objectContainer->CreateElementAt(componentnr) = this->CreateComponent(componentName);
  }
  catch (itk::ExceptionObject & excp)
  {
    xl::xout["error"] << "ERROR: error occurred while creating " << key << " " << componentnr << "." << std::endl;
    xl::xout["error"] << excp << std::endl;
    errorcode = 1;
    return objectContainer;
  }

  /** Check if more than one component name is given. */
  while (found)
  {
    ++componentnr;
    found = this->m_Configuration->ReadParameter(componentName, key, componentnr, false);
    if (found)
    {
      try
      {
        objectContainer->CreateElementAt(componentnr) = this->CreateComponent(componentName);
      }
      catch (itk::ExceptionObject & excp)
      {
        xl::xout["error"] << "ERROR: error occurred while creating " << key << " " << componentnr << "." << std::endl;
        xl::xout["error"] << excp << std::endl;
        errorcode = 1;
        return objectContainer;
      }
    } // end if
  }   // end while

  return objectContainer;

} // end CreateComponents()


/**
 * *********************** SetProcessPriority *************************
 */

void
ElastixMain::SetProcessPriority(void) const
{
  /** If wanted, set the priority of this process high or below normal. */
  std::string processPriority = this->m_Configuration->GetCommandLineArgument("-priority");
  if (processPriority == "high")
  {
#if defined(_WIN32) && !defined(__CYGWIN__)
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
  }
  else if (processPriority == "abovenormal")
  {
#if defined(_WIN32) && !defined(__CYGWIN__)
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
#endif
  }
  else if (processPriority == "normal")
  {
#if defined(_WIN32) && !defined(__CYGWIN__)
    SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
#endif
  }
  else if (processPriority == "belownormal")
  {
#if defined(_WIN32) && !defined(__CYGWIN__)
    SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
#endif
  }
  else if (processPriority == "idle")
  {
#if defined(_WIN32) && !defined(__CYGWIN__)
    SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);
#endif
  }
  else if (processPriority != "")
  {
    xl::xout["warning"]
      << "Unsupported -priority value. Specify one of <high, abovenormal, normal, belownormal, idle, ''>." << std::endl;
  }

} // end SetProcessPriority()


/**
 * *********************** SetMaximumNumberOfThreads *************************
 */

void
ElastixMain::SetMaximumNumberOfThreads(void) const
{
  /** Get the number of threads from the command line. */
  std::string maximumNumberOfThreadsString = this->m_Configuration->GetCommandLineArgument("-threads");

  /** If supplied, set the maximum number of threads. */
  if (maximumNumberOfThreadsString != "")
  {
    const int maximumNumberOfThreads = atoi(maximumNumberOfThreadsString.c_str());
    itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(maximumNumberOfThreads);
  }
} // end SetMaximumNumberOfThreads()


/**
 * ******************** SetOriginalFixedImageDirectionFlat ********************
 */

void
ElastixMain::SetOriginalFixedImageDirectionFlat(const FlatDirectionCosinesType & arg)
{
  this->m_OriginalFixedImageDirection = arg;
} // end SetOriginalFixedImageDirectionFlat()


/**
 * ******************** GetOriginalFixedImageDirectionFlat ********************
 */

const ElastixMain::FlatDirectionCosinesType &
ElastixMain::GetOriginalFixedImageDirectionFlat(void) const
{
  return this->m_OriginalFixedImageDirection;
} // end GetOriginalFixedImageDirectionFlat()


/**
 * ******************** GetTransformParametersMap ********************
 */

ElastixMain::ParameterMapType
ElastixMain::GetTransformParametersMap(void) const
{
  return this->m_TransformParametersMap;
} // end GetTransformParametersMap()


/**
 * ******************** GetImageInformationFromFile ********************
 */

void
ElastixMain::GetImageInformationFromFile(const std::string & filename, ImageDimensionType & imageDimension) const
{
  if (filename != "")
  {
    /** Dummy image type. */
    const unsigned int                                 DummyDimension = 3;
    typedef short                                      DummyPixelType;
    typedef itk::Image<DummyPixelType, DummyDimension> DummyImageType;

    /** Create a testReader. */
    typedef itk::ImageFileReader<DummyImageType> ReaderType;
    ReaderType::Pointer                          testReader = ReaderType::New();
    testReader->SetFileName(filename.c_str());

    /** Generate all information. */
    testReader->UpdateOutputInformation();

    /** Extract the required information. */
    itk::SmartPointer<const itk::ImageIOBase> testImageIO = testReader->GetImageIO();
    // itk::ImageIOBase::IOComponentType componentType = testImageIO->GetComponentType();
    // pixelType = itk::ImageIOBase::GetComponentTypeAsString( componentType );
    if (testImageIO.IsNull())
    {
      /** Extra check. In principal, ITK the testreader should already have thrown an exception
       * if it was not possible to create the ImageIO object */
      itkExceptionMacro(<< "ERROR: ImageIO object was not created, but no exception was thrown.");
    }
    imageDimension = testImageIO->GetNumberOfDimensions();
  } // end if

} // end GetImageInformationFromFile()


} // end namespace elastix
