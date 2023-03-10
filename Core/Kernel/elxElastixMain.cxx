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

#include "elxElastixMain.h"
#include "elxDeref.h"

#ifdef ELASTIX_USE_OPENCL
#  include "itkOpenCLContext.h"
#  include "itkOpenCLSetup.h"
#endif

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

ElastixMain::ElastixMain() = default;

/**
 * ********************** Destructor ****************************
 */

ElastixMain::~ElastixMain() = default;


/**
 * **************************** Run *****************************
 *
 * Assuming EnterCommandLineParameters has already been invoked.
 * or that m_Configuration is initialized in another way.
 */

int
ElastixMain::Run()
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
  catch (const itk::ExceptionObject & excp)
  {
    /** We just print the exception and let the program quit. */
    log::error(std::ostringstream{} << excp);
    errorCode = 1;
    return errorCode;
  }

  const Configuration & configuration = Deref(MainBase::GetConfiguration());

  /** Create OpenCL context and logger here. */
#ifdef ELASTIX_USE_OPENCL
  /** Check if user overrides OpenCL device selection. */
  std::string userSuppliedOpenCLDeviceType = "GPU";
  configuration.ReadParameter(userSuppliedOpenCLDeviceType, "OpenCLDeviceType", 0, false);

  int userSuppliedOpenCLDeviceID = -1;
  configuration.ReadParameter(userSuppliedOpenCLDeviceID, "OpenCLDeviceID", 0, false);

  std::string errorMessage = "";
  const bool  creatingContextSuccessful =
    itk::CreateOpenCLContext(errorMessage, userSuppliedOpenCLDeviceType, userSuppliedOpenCLDeviceID);
  if (!creatingContextSuccessful)
  {
    /** Report and disable the GPU by releasing the context. */
    log::info(std::ostringstream{} << errorMessage << '\n' << "  OpenCL processing in elastix is disabled.\n");

    itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
    context->Release();
  }

  /** Create a log file. */
  itk::CreateOpenCLLogger("elastix", configuration.GetCommandLineArgument("-out"));
#endif
  auto & elastixBase = this->GetElastixBase();

  /** Set some information in the ElastixBase. */
  elastixBase.SetConfiguration(MainBase::GetConfiguration());
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
    log::error("ERROR: One or more components could not be created.");
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
  catch (const itk::ExceptionObject & excp1)
  {
    /** We just print the itk::exception and let the program quit. */
    log::error(std::ostringstream{} << excp1);
    errorCode = 1;
  }
  catch (const std::exception & excp2)
  {
    /** We just print the std::exception and let the program quit. */
    log::error(std::ostringstream{} << "std: " << excp2.what());
    errorCode = 1;
  }
  catch (...)
  {
    /** We don't know what happened and just print a general message. */
    log::error(std::ostringstream{} << "ERROR: an unknown non-ITK, non-std exception was caught.\n"
                                    << "Please report this to elastix@bigr.nl.");
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
 * ************************** InitDBIndex ***********************
 *
 * Checks if the configuration object has been initialized,
 * determines the requested ImageTypes, and sets the m_DBIndex
 * to the corresponding value (by asking the elx::ComponentDatabase).
 */

int
ElastixMain::InitDBIndex()
{
  const Configuration & configuration = Deref(MainBase::GetConfiguration());

  /** Only do something when the configuration object wasn't initialized yet. */
  if (configuration.IsInitialized())
  {
    /** FixedImagePixelType. */
    if (this->m_FixedImagePixelType.empty())
    {
      /** Try to read it from the parameter file. */
      this->m_FixedImagePixelType = "float"; // \note: this assumes elastix was compiled for float
      configuration.ReadParameter(this->m_FixedImagePixelType, "FixedInternalImagePixelType", 0);
    }

    /** FixedImageDimension. */
    if (this->m_FixedImageDimension == 0)
    {
      if (!BaseComponent::IsElastixLibrary())
      {
        /** Get the fixed image file name. */
        std::string fixedImageFileName = configuration.GetCommandLineArgument("-f");
        if (fixedImageFileName.empty())
        {
          fixedImageFileName = configuration.GetCommandLineArgument("-f0");
        }

        /** Sanity check. */
        if (fixedImageFileName.empty())
        {
          log::error(std::ostringstream{} << "ERROR: could not read fixed image.\n"
                                          << "  both -f and -f0 are unspecified");
          return 1;
        }

        /** Read it from the fixed image header. */
        try
        {
          this->GetImageInformationFromFile(fixedImageFileName, this->m_FixedImageDimension);
        }
        catch (const itk::ExceptionObject & err)
        {
          log::error(std::ostringstream{} << "ERROR: could not read fixed image.\n" << err);
          return 1;
        }

        /** Try to read it from the parameter file.
         * This only serves as a check; elastix versions prior to 4.6 read the dimension
         * from the parameter file, but now we read it from the image header.
         */
        unsigned int fixDimParameterFile = 0;
        bool foundInParameterFile = configuration.ReadParameter(fixDimParameterFile, "FixedImageDimension", 0, false);

        /** Check. */
        if (foundInParameterFile)
        {
          if (fixDimParameterFile != this->m_FixedImageDimension)
          {
            log::error(std::ostringstream{}
                       << "ERROR: problem defining fixed image dimension.\n"
                       << "  The parameter file says:     " << fixDimParameterFile << "\n"
                       << "  The fixed image header says: " << this->m_FixedImageDimension << "\n"
                       << "  Note that from elastix 4.6 the parameter file definition \"FixedImageDimension\" "
                          "is not needed anymore.\n  Please remove this entry from your parameter file.");
            return 1;
          }
        }
      }
      else
      {
        configuration.ReadParameter(this->m_FixedImageDimension, "FixedImageDimension", 0, false);
      }

      /** Just a sanity check, probably not needed. */
      if (this->m_FixedImageDimension == 0)
      {
        log::error("ERROR: The FixedImageDimension is not given.");
        return 1;
      }
    }

    /** MovingImagePixelType. */
    if (this->m_MovingImagePixelType.empty())
    {
      /** Try to read it from the parameter file. */
      this->m_MovingImagePixelType = "float"; // \note: this assumes elastix was compiled for float
      configuration.ReadParameter(this->m_MovingImagePixelType, "MovingInternalImagePixelType", 0);
    }

    /** MovingImageDimension. */
    if (this->m_MovingImageDimension == 0)
    {
      if (!BaseComponent::IsElastixLibrary())
      {
        /** Get the moving image file name. */
        std::string movingImageFileName = configuration.GetCommandLineArgument("-m");
        if (movingImageFileName.empty())
        {
          movingImageFileName = configuration.GetCommandLineArgument("-m0");
        }

        /** Sanity check. */
        if (movingImageFileName.empty())
        {
          log::error(std::ostringstream{} << "ERROR: could not read moving image.\n"
                                          << "  both -m and -m0 are unspecified");
          return 1;
        }

        /** Read it from the moving image header. */
        try
        {
          this->GetImageInformationFromFile(movingImageFileName, this->m_MovingImageDimension);
        }
        catch (const itk::ExceptionObject & err)
        {
          log::error(std::ostringstream{} << "ERROR: could not read moving image.\n" << err);
          return 1;
        }

        /** Try to read it from the parameter file.
         * This only serves as a check; elastix versions prior to 4.6 read the dimension
         * from the parameter file, but now we read it from the image header.
         */
        unsigned int movDimParameterFile = 0;
        bool foundInParameterFile = configuration.ReadParameter(movDimParameterFile, "MovingImageDimension", 0, false);

        /** Check. */
        if (foundInParameterFile)
        {
          if (movDimParameterFile != this->m_MovingImageDimension)
          {
            log::error(std::ostringstream{}
                       << "ERROR: problem defining moving image dimension.\n"
                       << "  The parameter file says:      " << movDimParameterFile << "\n"
                       << "  The moving image header says: " << this->m_MovingImageDimension << "\n"
                       << "  Note that from elastix 4.6 the parameter file definition \"MovingImageDimension\" "
                          "is not needed anymore.\n  Please remove this entry from your parameter file.");
            return 1;
          }
        }
      }
      else
      {
        configuration.ReadParameter(this->m_MovingImageDimension, "MovingImageDimension", 0, false);
      }

      /** Just a sanity check, probably not needed. */
      if (this->m_MovingImageDimension == 0)
      {
        log::error("ERROR: The MovingImageDimension is not given.");
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
      log::error("ERROR: Something went wrong in the ComponentDatabase");
      return 1;
    }

  } // end if configuration.Initialized();
  else
  {
    log::error("ERROR: The configuration object has not been initialized.");
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
  Configuration & configuration = Deref(MainBase::GetConfiguration());
  configuration.SetElastixLevel(level);

} // end SetElastixLevel()


/**
 * ********************* GetElastixLevel ************************
 */

unsigned int
ElastixMain::GetElastixLevel() const
{
  /** Call GetElastixLevel from MyConfiguration. */
  const Configuration & configuration = Deref(MainBase::GetConfiguration());
  return configuration.GetElastixLevel();

} // end GetElastixLevel()


/**
 * ********************* SetTotalNumberOfElastixLevels ************************
 */

void
ElastixMain::SetTotalNumberOfElastixLevels(unsigned int levels)
{
  /** Call SetTotalNumberOfElastixLevels from MyConfiguration. */
  Configuration & configuration = Deref(MainBase::GetConfiguration());
  configuration.SetTotalNumberOfElastixLevels(levels);

} // end SetTotalNumberOfElastixLevels()


/**
 * ********************* GetTotalNumberOfElastixLevels ************************
 */

unsigned int
ElastixMain::GetTotalNumberOfElastixLevels() const
{
  /** Call GetTotalNumberOfElastixLevels from MyConfiguration. */
  const Configuration & configuration = Deref(MainBase::GetConfiguration());
  return configuration.GetTotalNumberOfElastixLevels();

} // end GetTotalNumberOfElastixLevels()


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
ElastixMain::GetOriginalFixedImageDirectionFlat() const
{
  return this->m_OriginalFixedImageDirection;
} // end GetOriginalFixedImageDirectionFlat()


/**
 * ******************** GetTransformParametersMap ********************
 */

ElastixMain::ParameterMapType
ElastixMain::GetTransformParametersMap() const
{
  return this->m_TransformParametersMap;
} // end GetTransformParametersMap()


/**
 * ******************** GetImageInformationFromFile ********************
 */

void
ElastixMain::GetImageInformationFromFile(const std::string & filename, ImageDimensionType & imageDimension) const
{
  if (!filename.empty())
  {
    /** Dummy image type. */
    const unsigned int DummyDimension = 3;
    using DummyPixelType = short;
    using DummyImageType = itk::Image<DummyPixelType, DummyDimension>;

    /** Create a testReader. */
    using ReaderType = itk::ImageFileReader<DummyImageType>;
    auto testReader = ReaderType::New();
    testReader->SetFileName(filename);

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
