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
#ifndef elxElastixTemplate_hxx
#  define elxElastixTemplate_hxx

#  include "elxElastixTemplate.h"
#  include <itkDeref.h>

#  define elxCheckAndSetComponentMacro(_name)                                                                         \
    _name##BaseType * base = this->GetElx##_name##Base(i);                                                            \
    if (base != nullptr)                                                                                              \
    {                                                                                                                 \
      base->SetComponentLabel(#_name, i);                                                                             \
      base->SetElastix(This);                                                                                         \
    }                                                                                                                 \
    else                                                                                                              \
    {                                                                                                                 \
      std::string par = "";                                                                                           \
      itk::Deref(ElastixBase::GetConfiguration()).ReadParameter(par, #_name, i, false);                               \
      itkExceptionMacro("ERROR: entry " << i << " of " << #_name << " reads \"" << par << "\", which is not of type " \
                                        << #_name << "BaseType.");                                                    \
    }
// end elxCheckAndSetComponentMacro

namespace elastix
{
/**
 * ********************** GetFixedImage *************************
 */

template <typename TFixedImage, typename TMovingImage>
auto
ElastixTemplate<TFixedImage, TMovingImage>::GetFixedImage(unsigned int idx) const -> FixedImageType *
{
  if (idx < this->GetNumberOfFixedImages())
  {
    return dynamic_cast<FixedImageType *>(this->GetFixedImageContainer()->ElementAt(idx).GetPointer());
  }

  return nullptr;

} // end GetFixedImage()

/**
 * ********************** GetMovingImage *************************
 */

template <typename TFixedImage, typename TMovingImage>
auto
ElastixTemplate<TFixedImage, TMovingImage>::GetMovingImage(unsigned int idx) const -> MovingImageType *
{
  if (idx < this->GetNumberOfMovingImages())
  {
    return dynamic_cast<MovingImageType *>(this->GetMovingImageContainer()->ElementAt(idx).GetPointer());
  }

  return nullptr;

} // end SetMovingImage()

/**
 * ********************** GetFixedMask *************************
 */

template <typename TFixedImage, typename TMovingImage>
auto
ElastixTemplate<TFixedImage, TMovingImage>::GetFixedMask(unsigned int idx) const -> FixedMaskType *
{
  if (idx < this->GetNumberOfFixedMasks())
  {
    return dynamic_cast<FixedMaskType *>(this->GetFixedMaskContainer()->ElementAt(idx).GetPointer());
  }

  return nullptr;

} // end SetFixedMask()

/**
 * ********************** GetMovingMask *************************
 */

template <typename TFixedImage, typename TMovingImage>
auto
ElastixTemplate<TFixedImage, TMovingImage>::GetMovingMask(unsigned int idx) const -> MovingMaskType *
{
  if (idx < this->GetNumberOfMovingMasks())
  {
    return dynamic_cast<MovingMaskType *>(this->GetMovingMaskContainer()->ElementAt(idx).GetPointer());
  }

  return nullptr;

} // end SetMovingMask()


/**
 * **************************** Run *****************************
 */

template <typename TFixedImage, typename TMovingImage>
int
ElastixTemplate<TFixedImage, TMovingImage>::Run()
{
  /** Tell all components where to find the ElastixTemplate and
   * set there ComponentLabel.
   */
  this->ConfigureComponents(this);

  /** Call BeforeAll to do some checking. */
  int dummy = this->BeforeAll();
  if (dummy != 0)
  {
    return dummy;
  }

  /** Setup Callbacks. This makes sure that the BeforeEachResolution()
   * and AfterEachIteration() functions are called.
   *
   * NB: it is not yet clear what should happen when multiple registration
   * or optimizer components are used simultaneously. We won't use this
   * in the near future anyway, probably.
   */
  m_BeforeEachResolutionCommand = BeforeEachResolutionCommandType::New();
  m_AfterEachResolutionCommand = AfterEachResolutionCommandType::New();
  m_AfterEachIterationCommand = AfterEachIterationCommandType::New();

  m_BeforeEachResolutionCommand->SetCallbackFunction(this, &Self::BeforeEachResolution);
  m_AfterEachResolutionCommand->SetCallbackFunction(this, &Self::AfterEachResolution);
  m_AfterEachIterationCommand->SetCallbackFunction(this, &Self::AfterEachIteration);

  this->GetElxRegistrationBase()->GetAsITKBaseType()->AddObserver(itk::IterationEvent(), m_BeforeEachResolutionCommand);
  this->GetElxOptimizerBase()->GetAsITKBaseType()->AddObserver(itk::IterationEvent(), m_AfterEachIterationCommand);
  this->GetElxOptimizerBase()->GetAsITKBaseType()->AddObserver(itk::EndEvent(), m_AfterEachResolutionCommand);

  /** Start the timer for reading images. */
  ElastixBase::m_Timer0.Start();
  log::info("\nReading images...");

  /** Read images and masks, if not set already. */
  const bool              useDirCos = this->GetUseDirectionCosines();
  FixedImageDirectionType fixDirCos;
  if (this->GetFixedImage() == nullptr)
  {
    this->SetFixedImageContainer(MultipleImageLoader<FixedImageType>::GenerateImageContainer(
      this->GetFixedImageFileNameContainer(), "Fixed Image", useDirCos, &fixDirCos));
    this->SetOriginalFixedImageDirection(fixDirCos);
  }
  else
  {
    /**
     *  images are set in elastixlib.cxx
     *  just set direction cosines
     *  in case images are imported for executable it does not matter
     *  because the InfoChanger has changed these images.
     */
    FixedImageType * fixedIm = this->GetFixedImage(0);
    fixDirCos = fixedIm->GetDirection();
    this->SetOriginalFixedImageDirection(fixDirCos);
  }

  if (this->GetMovingImage() == nullptr)
  {
    this->SetMovingImageContainer(MultipleImageLoader<MovingImageType>::GenerateImageContainer(
      this->GetMovingImageFileNameContainer(), "Moving Image", useDirCos));
  }
  if (this->GetFixedMask() == nullptr)
  {
    this->SetFixedMaskContainer(MultipleImageLoader<FixedMaskType>::GenerateImageContainer(
      this->GetFixedMaskFileNameContainer(), "Fixed Mask", useDirCos));
  }
  if (this->GetMovingMask() == nullptr)
  {
    this->SetMovingMaskContainer(MultipleImageLoader<MovingMaskType>::GenerateImageContainer(
      this->GetMovingMaskFileNameContainer(), "Moving Mask", useDirCos));
  }

  /** Print the time spent on reading images. */
  ElastixBase::m_Timer0.Stop();
  log::info(std::ostringstream{} << "Reading images took "
                                 << static_cast<std::uint64_t>(ElastixBase::m_Timer0.GetMean() * 1000) << " ms.\n");

  /** Give all components the opportunity to do some initialization. */
  this->BeforeRegistration();

  /** START! */
  try
  {
    (this->GetElxRegistrationBase()->GetAsITKBaseType())->StartRegistration();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("ElastixTemplate - Run()");
    std::string err_str = excp.GetDescription();
    err_str += "\n\nError occurred during actual registration.";
    excp.SetDescription(err_str);

    /** Pass the exception to a higher level. */
    throw;
  }

  /** Save, show results etc. */
  this->AfterRegistration();

  /** Make sure that the transform has stored the final parameters.
   *
   * The transform may be used as a transform in a next elastixLevel;
   * We need to be sure that it has the final parameters set.
   * In the AfterRegistration-method of TransformBase, this method is
   * already called, but some other component may change the parameters
   * again in its AfterRegistration-method.
   *
   * For now we leave it commented, since there is only Resampler, which
   * already calls this method. Calling it again would just take time.
   */
  // this->GetElxTransformBase()->SetFinalParameters();

  /** Set the first transform as the final transform. This means that
   * the other transforms should be set as an initial transform of this
   * transforms. However, up to now, multiple transforms are not really
   * supported yet.
   */
  this->SetFinalTransform(this->GetTransformContainer()->ElementAt(0));

  /** Decouple the components from Elastix. This increases the chance that
   * some memory is released.
   */
  this->ConfigureComponents(nullptr);

  /** Return a value. */
  return 0;

} // end Run()


/**
 * ************************ ApplyTransform **********************
 */

template <typename TFixedImage, typename TMovingImage>
int
ElastixTemplate<TFixedImage, TMovingImage>::ApplyTransform(const bool doReadTransform)
{
  /** Timer. */
  itk::TimeProbe timer;

  /** Tell all components where to find the ElastixTemplate. */
  this->ConfigureComponents(this);

  /** Call BeforeAllTransformix to do some checking. */
  int dummy = this->BeforeAllTransformix();
  if (dummy != 0)
  {
    return dummy;
  }

  /** Set the inputImage (=movingImage).
   * If "-in" was given or an input image was given in some other way,
   * load the image.
   */
  if ((this->GetNumberOfMovingImageFileNames() > 0) || (this->GetMovingImage() != nullptr))
  {
    /** Timer. */
    timer.Start();

    /** Tell the user. */
    log::info(std::ostringstream{} << '\n' << "Reading input image ...");

    /** Load the image from disk, if it wasn't set already by the user. */
    const bool useDirCos = this->GetUseDirectionCosines();
    if (this->GetMovingImage() == nullptr)
    {
      this->SetMovingImageContainer(MultipleImageLoader<MovingImageType>::GenerateImageContainer(
        this->GetMovingImageFileNameContainer(), "Input Image", useDirCos));
    } // end if !moving image

    /** Tell the user. */
    timer.Stop();
    log::info(std::ostringstream{} << "  Reading input image took " << timer.GetMean() << " s");

  } // end if inputImageFileName

  /** Call all the ReadFromFile() functions. */
  timer.Reset();
  timer.Start();
  log::info("Calling all ReadFromFile()'s ...");

  this->GetElxResampleInterpolatorBase()->ReadFromFile();

  auto & elxResamplerBase = *(this->GetElxResamplerBase());
  auto & elxTransformBase = *(this->GetElxTransformBase());

  elxResamplerBase.ReadFromFile();

  if (doReadTransform)
  {
    elxTransformBase.ReadFromFile();
  }

  /** Tell the user. */
  timer.Stop();
  log::info(std::ostringstream{} << "  Calling all ReadFromFile()'s took " << timer.GetMean() << " s");

  /** Call TransformPoints.
   * Actually we could loop over all transforms.
   * But for now, there seems to be no use yet for that.
   */
  timer.Reset();
  timer.Start();
  log::info("Transforming points ...");
  try
  {
    elxTransformBase.TransformPoints();
  }
  catch (const itk::ExceptionObject & excp)
  {
    log::error(std::ostringstream{} << excp << '\n' << "However, transformix continues anyway.");
  }
  timer.Stop();
  log::info(std::ostringstream{} << "  Transforming points done, it took "
                                 << Conversion::SecondsToDHMS(timer.GetMean(), 2));

  /** Call ComputeSpatialJacobianDeterminantImage.
   * Actually we could loop over all transforms.
   * But for now, there seems to be no use yet for that.
   */
  timer.Reset();
  timer.Start();
  log::info("Compute determinant of spatial Jacobian ...");
  try
  {
    elxTransformBase.ComputeAndWriteSpatialJacobianDeterminantImage();
  }
  catch (const itk::ExceptionObject & excp)
  {
    log::error(std::ostringstream{} << excp << '\n' << "However, transformix continues anyway.");
  }
  timer.Stop();
  log::info(std::ostringstream{} << "  Computing determinant of spatial Jacobian done, it took "
                                 << Conversion::SecondsToDHMS(timer.GetMean(), 2));

  /** Call ComputeAndWriteSpatialJacobianMatrixImage.
   * Actually we could loop over all transforms.
   * But for now, there seems to be no use yet for that.
   */
  timer.Reset();
  timer.Start();
  log::info("Compute spatial Jacobian (full matrix) ...");
  try
  {
    elxTransformBase.ComputeAndWriteSpatialJacobianMatrixImage();
  }
  catch (const itk::ExceptionObject & excp)
  {
    log::error(std::ostringstream{} << excp << '\n' << "However, transformix continues anyway.");
  }
  timer.Stop();
  log::info(std::ostringstream{} << "  Computing spatial Jacobian done, it took "
                                 << Conversion::SecondsToDHMS(timer.GetMean(), 2));

  /** Resample the image. */
  if (this->GetMovingImage() != nullptr)
  {
    timer.Reset();
    timer.Start();
    log::info("Resampling image and writing to disk ...");

    /** Write the resampled image to disk.
     * Actually we could loop over all resamplers.
     * But for now, there seems to be no use yet for that.
     */
    if (!BaseComponent::IsElastixLibrary())
    {
      // It is assumed the configuration is not null at this point in time.
      const Configuration & configuration = itk::Deref(ElastixBase::GetConfiguration());

      /** Create a name for the final result. */
      const auto  resultImageName = configuration.RetrieveParameterStringValue("result", "ResultImageName", 0, false);
      std::string resultImageFormat = "mhd";
      configuration.ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);
      std::ostringstream makeFileName;
      makeFileName << configuration.GetCommandLineArgument("-out") << resultImageName << '.' << resultImageFormat;

      elxResamplerBase.ResampleAndWriteResultImage(makeFileName.str(), true);
    }
    else
    {
      elxResamplerBase.CreateItkResultImage();
    }

    /** Print the elapsed time for the resampling. */
    timer.Stop();
    log::info(std::ostringstream{} << "  Resampling took " << Conversion::SecondsToDHMS(timer.GetMean(), 2));
  }

  /** Return a value. */
  return 0;

} // end ApplyTransform()


/**
 * ************************ BeforeAll ***************************
 */

template <typename TFixedImage, typename TMovingImage>
int
ElastixTemplate<TFixedImage, TMovingImage>::BeforeAll()
{
  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Call all the BeforeRegistration() functions. */
  returndummy |= this->BeforeAllBase();
  returndummy |= CallInEachComponentInt(&BaseComponentType::BeforeAllBase);
  returndummy |= CallInEachComponentInt(&BaseComponentType::BeforeAll);

  /** Return a value. */
  return returndummy;

} // end BeforeAll()


/**
 * ******************** BeforeAllTransformix ********************
 */

template <typename TFixedImage, typename TMovingImage>
int
ElastixTemplate<TFixedImage, TMovingImage>::BeforeAllTransformix()
{
  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Call the BeforeAllTransformixBase function in ElastixBase.
   * It checks most of the parameters. For now, it is the only
   * component that has a BeforeAllTranformixBase() method.
   */
  returndummy |= this->BeforeAllTransformixBase();

  /** Call all the BeforeAllTransformix() functions.
   * Actually we could loop over all resample interpolators, resamplers,
   * and transforms etc. But for now, there seems to be no use yet for that.
   */
  returndummy |= this->GetElxResampleInterpolatorBase()->BeforeAllTransformix();
  returndummy |= this->GetElxResamplerBase()->BeforeAllTransformix();
  returndummy |= this->GetElxTransformBase()->BeforeAllTransformix();

  /** The GetConfiguration also has a BeforeAllTransformix,
   * It print the Transform Parameter file to the log file. That's
   * why we call it after the other components.
   */
  if (!BaseComponent::IsElastixLibrary())
  {
    returndummy |= this->GetConfiguration()->BeforeAllTransformix();
  }

  /** Return a value. */
  return returndummy;

} // end BeforeAllTransformix()


/**
 * **************** BeforeRegistration *****************
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::BeforeRegistration()
{
  /** Start timer for initializing all components. */
  ElastixBase::m_Timer0.Reset();
  ElastixBase::m_Timer0.Start();

  /** Call all the BeforeRegistration() functions. */
  this->BeforeRegistrationBase();
  CallInEachComponent(&BaseComponentType::BeforeRegistrationBase);
  CallInEachComponent(&BaseComponentType::BeforeRegistration);

  /** Add a column to iteration with the iteration number. */
  this->AddTargetCellToIterationInfo("1:ItNr");

  /** Add a column to iteration with timing information. */
  this->AddTargetCellToIterationInfo("Time[ms]");
  this->GetIterationInfoAt("Time[ms]") << std::showpoint << std::fixed << std::setprecision(1);

  /** Print time for initializing. */
  ElastixBase::m_Timer0.Stop();
  log::info(std::ostringstream{} << "Initialization of all components (before registration) took: "
                                 << static_cast<std::uint64_t>(ElastixBase::m_Timer0.GetMean() * 1000) << " ms.");

  /** Start Timer0 here, to make it possible to measure the time needed for
   * preparation of the first resolution.
   */
  ElastixBase::m_Timer0.Reset();
  ElastixBase::m_Timer0.Start();

} // end BeforeRegistration()


/**
 * ************** BeforeEachResolution *****************
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::BeforeEachResolution()
{
  /** Get current resolution level. */
  unsigned long level = this->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel();

  if (level == 0)
  {
    ElastixBase::m_Timer0.Stop();
    log::info(std::ostringstream{} << "Preparation of the image pyramids took: "
                                   << static_cast<std::uint64_t>(ElastixBase::m_Timer0.GetMean() * 1000) << " ms.");
    ElastixBase::m_Timer0.Reset();
    ElastixBase::m_Timer0.Start();
  }

  /** Reset the ElastixBase::m_IterationCounter. */
  ElastixBase::m_IterationCounter = 0;

  /** Print the current resolution. */
  log::info(std::ostringstream{} << "\nResolution: " << level);

  const Configuration & configuration = itk::Deref(ElastixBase::GetConfiguration());

  /** Create a TransformParameter-file for the current resolution. */
  bool writeIterationInfo = true;
  configuration.ReadParameter(writeIterationInfo, "WriteIterationInfo", 0, false);
  if (writeIterationInfo)
  {
    this->OpenIterationInfoFile();
  }

  /** Call all the BeforeEachResolution() functions. */
  this->BeforeEachResolutionBase();
  CallInEachComponent(&BaseComponentType::BeforeEachResolutionBase);
  CallInEachComponent(&BaseComponentType::BeforeEachResolution);

  /** Print the extra preparation time needed for this resolution. */
  ElastixBase::m_Timer0.Stop();
  log::info(std::ostringstream{} << "Elastix initialization of all components (for this resolution) took: "
                                 << static_cast<std::uint64_t>(ElastixBase::m_Timer0.GetMean() * 1000) << " ms.");

  /** Start ResolutionTimer, which measures the total iteration time in this resolution. */
  ElastixBase::m_ResolutionTimer.Reset();
  ElastixBase::m_ResolutionTimer.Start();

  /** Start IterationTimer here, to make it possible to measure the time
   * of the first iteration.
   */
  ElastixBase::m_IterationTimer.Reset();
  ElastixBase::m_IterationTimer.Start();

} // end BeforeEachResolution()


/**
 * ************** AfterEachResolution *****************
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::AfterEachResolution()
{
  /** Get current resolution level. */
  unsigned long level = this->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel();

  /** Print the total iteration time. */
  ElastixBase::m_ResolutionTimer.Stop();
  log::info(std::ostringstream{} << std::setprecision(3) << "Time spent in resolution " << (level)
                                 << " (ITK initialization and iterating): "
                                 << ElastixBase::m_ResolutionTimer.GetMean());

  /** Call all the AfterEachResolution() functions. */
  this->AfterEachResolutionBase();
  CallInEachComponent(&BaseComponentType::AfterEachResolutionBase);
  CallInEachComponent(&BaseComponentType::AfterEachResolution);

  const Configuration & configuration = itk::Deref(ElastixBase::GetConfiguration());

  /** Create a TransformParameter-file for the current resolution. */
  bool writeTransformParameterEachResolution = false;
  configuration.ReadParameter(
    writeTransformParameterEachResolution, "WriteTransformParametersEachResolution", 0, false);

  const std::string outputDirectoryPath = configuration.GetCommandLineArgument("-out");

  if (writeTransformParameterEachResolution && !outputDirectoryPath.empty())
  {
    /** Create the TransformParameters filename for this resolution. */
    std::ostringstream makeFileName;
    makeFileName << outputDirectoryPath << "TransformParameters." << configuration.GetElastixLevel() << ".R"
                 << this->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel() << ".txt";
    std::string fileName = makeFileName.str();

    /** Create a TransformParameterFile for this iteration. */
    this->CreateTransformParameterFile(fileName, false);
  }

  /** Start Timer0 here, to make it possible to measure the time needed for:
   *    - executing the BeforeEachResolution methods (if this was not the last resolution)
   *    - executing the AfterRegistration methods (if this was the last resolution)
   */
  ElastixBase::m_Timer0.Reset();
  ElastixBase::m_Timer0.Start();

} // end AfterEachResolution()


/**
 * ************** AfterEachIteration *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::AfterEachIteration()
{
  /** Write the headers of the columns that are printed each iteration. */
  if (ElastixBase::m_IterationCounter == 0)
  {
    this->GetIterationInfo().WriteHeaders();
  }

  /** Call all the AfterEachIteration() functions. */
  this->AfterEachIterationBase();
  CallInEachComponent(&BaseComponentType::AfterEachIterationBase);
  CallInEachComponent(&BaseComponentType::AfterEachIteration);

  /** Write the iteration number to the table. */
  this->GetIterationInfoAt("1:ItNr") << m_IterationCounter;

  /** Time in this iteration. */
  ElastixBase::m_IterationTimer.Stop();
  this->GetIterationInfoAt("Time[ms]") << ElastixBase::m_IterationTimer.GetMean() * 1000.0;

  /** Write the iteration info of this iteration. */
  this->GetIterationInfo().WriteBufferedData();

  const Configuration & configuration = itk::Deref(ElastixBase::GetConfiguration());

  /** Create a TransformParameter-file for the current iteration. */
  bool writeTansformParametersThisIteration = false;
  configuration.ReadParameter(writeTansformParametersThisIteration, "WriteTransformParametersEachIteration", 0, false);

  const std::string outputDirectoryPath = configuration.GetCommandLineArgument("-out");

  if (writeTansformParametersThisIteration && !outputDirectoryPath.empty())
  {
    /** Add zeros to the number of iterations, to make sure
     * it always consists of 7 digits.
     * \todo: use sprintf for this. it's much easier. or a formatting string for the
     * ostringstream, if that's possible somehow.
     */
    std::ostringstream makeIterationString;
    unsigned int       border = 1000000;
    while (border > 1)
    {
      if (ElastixBase::m_IterationCounter < border)
      {
        makeIterationString << "0";
        border /= 10;
      }
      else
      {
        /** Stop. */
        border = 1;
      }
    }
    makeIterationString << ElastixBase::m_IterationCounter;

    /** Create the TransformParameters filename for this iteration. */
    std::ostringstream makeFileName;
    makeFileName << outputDirectoryPath << "TransformParameters." << configuration.GetElastixLevel() << ".R"
                 << this->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel() << ".It"
                 << makeIterationString.str() << ".txt";
    std::string tpFileName = makeFileName.str();

    /** Create a TransformParameterFile for this iteration. */
    this->CreateTransformParameterFile(tpFileName, false);
  }

  /** Count the number of iterations. */
  ElastixBase::m_IterationCounter++;

  /** Start timer for next iteration. */
  ElastixBase::m_IterationTimer.Reset();
  ElastixBase::m_IterationTimer.Start();

} // end AfterEachIteration()


/**
 * ************** AfterRegistration *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::AfterRegistration()
{
  itk::TimeProbe timer;
  timer.Start();

  /** A white line. */
  elx::log::info("");

  const Configuration & configuration = itk::Deref(ElastixBase::GetConfiguration());

  /** Create the final TransformParameters filename. */
  bool writeFinalTansformParameters = true;
  configuration.ReadParameter(writeFinalTansformParameters, "WriteFinalTransformParameters", 0, false);

  const std::string outputDirectoryPath = configuration.GetCommandLineArgument("-out");

  if (writeFinalTansformParameters && !outputDirectoryPath.empty())
  {
    const std::string parameterMapFileFormat =
      configuration.RetrieveParameterStringValue("", "OutputTransformParameterFileFormat", 0, false);
    const auto format = Conversion::StringToParameterMapStringFormat(parameterMapFileFormat);

    std::ostringstream makeFileName;
    makeFileName << outputDirectoryPath << "TransformParameters." << configuration.GetElastixLevel()
                 << Conversion::CreateParameterMapFileNameExtension(format);
    std::string FileName = makeFileName.str();

    /** Create a final TransformParameterFile. */
    this->CreateTransformParameterFile(FileName, true);
  }

  if (BaseComponent::IsElastixLibrary())
  {
    /** Get the transform parameters. */
    this->CreateTransformParameterMap(); // only relevant for dll!
  }

  timer.Stop();
  log::info(std::ostringstream{} << "\nCreating the TransformParameterFile took "
                                 << Conversion::SecondsToDHMS(timer.GetMean(), 2));

  /** Call all the AfterRegistration() functions. */
  this->AfterRegistrationBase();
  CallInEachComponent(&BaseComponentType::AfterRegistrationBase);
  CallInEachComponent(&BaseComponentType::AfterRegistration);

  /** Print the time spent on things after the registration. */
  ElastixBase::m_Timer0.Stop();
  log::info(std::ostringstream{} << "Time spent on saving the results, applying the final transform etc.: "
                                 << static_cast<std::uint64_t>(ElastixBase::m_Timer0.GetMean() * 1000) << " ms.");

} // end AfterRegistration()


/**
 * ************** CreateTransformParameterFile ******************
 *
 * Setup the transform parameter file, which will
 * contain the final transform parameters.
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::CreateTransformParameterFile(const std::string & fileName, const bool toLog)
{
  /** Store CurrentTransformParameterFileName. */
  ElastixBase::m_CurrentTransformParameterFileName = fileName;

  /** Set it in the Transform, for later use. */
  this->GetElxTransformBase()->SetTransformParameterFileName(fileName);

  /** Separate clearly in log-file. */
  if (toLog)
  {
    log::info_to_log_file("\n=============== start of TransformParameterFile ===============");
  }

  /** Create transformationParameterInfo. */
  std::ostringstream transformationParameterInfo;

  /** Call all the WriteToFile() functions.
   * Actually we could loop over all resample interpolators, resamplers,
   * and transforms etc. But for now, there seems to be no use yet for that.
   */
  this->GetElxTransformBase()->WriteToFile(transformationParameterInfo,
                                           this->GetElxOptimizerBase()->GetAsITKBaseType()->GetCurrentPosition());
  this->GetElxResampleInterpolatorBase()->WriteToFile(transformationParameterInfo);
  this->GetElxResamplerBase()->WriteToFile(transformationParameterInfo);

  std::ofstream transformParameterFile(fileName);

  if (transformParameterFile.is_open())
  {
    transformParameterFile << transformationParameterInfo.str();
  }
  else
  {
    log::error(std::ostringstream{} << "ERROR: File \"" << fileName << "\" could not be opened!");
  }

  /** Separate clearly in log-file. */
  if (toLog)
  {
    log::info_to_log_file(transformationParameterInfo.str());
    log::info_to_log_file("=============== end of TransformParameterFile ===============");
  }

} // end CreateTransformParameterFile()


/**
 * ************** CreateTransformParameterMap ******************
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::CreateTransformParameterMap()
{
  this->GetElxTransformBase()->CreateTransformParameterMap(
    this->GetElxOptimizerBase()->GetAsITKBaseType()->GetCurrentPosition(), ElastixBase::m_TransformParameterMap);
  this->GetElxResampleInterpolatorBase()->CreateTransformParameterMap(ElastixBase::m_TransformParameterMap);
  this->GetElxResamplerBase()->CreateTransformParameterMap(ElastixBase::m_TransformParameterMap);

} // end CreateTransformParameterMap()


/**
 * ****************** CallInEachComponent ***********************
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::CallInEachComponent(PtrToMemberFunction func)
{
  /** Call the memberfunction 'func' of all components. */
  ((*(this->GetConfiguration())).*func)();

  for (unsigned int i = 0; i < this->GetNumberOfRegistrations(); ++i)
  {
    ((*(this->GetElxRegistrationBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfTransforms(); ++i)
  {
    ((*(this->GetElxTransformBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfImageSamplers(); ++i)
  {
    ((*(this->GetElxImageSamplerBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfMetrics(); ++i)
  {
    ((*(this->GetElxMetricBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfInterpolators(); ++i)
  {
    ((*(this->GetElxInterpolatorBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfOptimizers(); ++i)
  {
    ((*(this->GetElxOptimizerBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfFixedImagePyramids(); ++i)
  {
    ((*(this->GetElxFixedImagePyramidBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfMovingImagePyramids(); ++i)
  {
    ((*(this->GetElxMovingImagePyramidBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfResampleInterpolators(); ++i)
  {
    ((*(this->GetElxResampleInterpolatorBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfResamplers(); ++i)
  {
    ((*(this->GetElxResamplerBase(i))).*func)();
  }

} // end CallInEachComponent()


/**
 * ****************** CallInEachComponentInt ********************
 */

template <typename TFixedImage, typename TMovingImage>
int
ElastixTemplate<TFixedImage, TMovingImage>::CallInEachComponentInt(PtrToMemberFunction2 func)
{
  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Call the memberfunction 'func' of all components. */
  returndummy |= ((*(this->GetConfiguration())).*func)();

  for (unsigned int i = 0; i < this->GetNumberOfRegistrations(); ++i)
  {
    returndummy |= ((*(this->GetElxRegistrationBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfTransforms(); ++i)
  {
    returndummy |= ((*(this->GetElxTransformBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfImageSamplers(); ++i)
  {
    returndummy |= ((*(this->GetElxImageSamplerBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfMetrics(); ++i)
  {
    returndummy |= ((*(this->GetElxMetricBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfInterpolators(); ++i)
  {
    returndummy |= ((*(this->GetElxInterpolatorBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfOptimizers(); ++i)
  {
    returndummy |= ((*(this->GetElxOptimizerBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfFixedImagePyramids(); ++i)
  {
    returndummy |= ((*(this->GetElxFixedImagePyramidBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfMovingImagePyramids(); ++i)
  {
    returndummy |= ((*(this->GetElxMovingImagePyramidBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfResampleInterpolators(); ++i)
  {
    returndummy |= ((*(this->GetElxResampleInterpolatorBase(i))).*func)();
  }
  for (unsigned int i = 0; i < this->GetNumberOfResamplers(); ++i)
  {
    returndummy |= ((*(this->GetElxResamplerBase(i))).*func)();
  }

  /** Return a value. */
  return returndummy;

} // end CallInEachComponent()


/**
 * ****************** ConfigureComponents *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::ConfigureComponents(Self * This)
{
  this->GetConfiguration()->SetComponentLabel("Configuration", 0);

  for (unsigned int i = 0; i < this->GetNumberOfRegistrations(); ++i)
  {
    elxCheckAndSetComponentMacro(Registration);
  }

  for (unsigned int i = 0; i < this->GetNumberOfTransforms(); ++i)
  {
    elxCheckAndSetComponentMacro(Transform);
  }

  for (unsigned int i = 0; i < this->GetNumberOfImageSamplers(); ++i)
  {
    elxCheckAndSetComponentMacro(ImageSampler);
  }

  for (unsigned int i = 0; i < this->GetNumberOfMetrics(); ++i)
  {
    elxCheckAndSetComponentMacro(Metric);
  }

  for (unsigned int i = 0; i < this->GetNumberOfInterpolators(); ++i)
  {
    elxCheckAndSetComponentMacro(Interpolator);
  }

  for (unsigned int i = 0; i < this->GetNumberOfOptimizers(); ++i)
  {
    elxCheckAndSetComponentMacro(Optimizer);
  }

  for (unsigned int i = 0; i < this->GetNumberOfFixedImagePyramids(); ++i)
  {
    elxCheckAndSetComponentMacro(FixedImagePyramid);
  }

  for (unsigned int i = 0; i < this->GetNumberOfMovingImagePyramids(); ++i)
  {
    elxCheckAndSetComponentMacro(MovingImagePyramid);
  }

  for (unsigned int i = 0; i < this->GetNumberOfResampleInterpolators(); ++i)
  {
    elxCheckAndSetComponentMacro(ResampleInterpolator);
  }

  for (unsigned int i = 0; i < this->GetNumberOfResamplers(); ++i)
  {
    elxCheckAndSetComponentMacro(Resampler);
  }

} // end ConfigureComponents()


/**
 * ************** OpenIterationInfoFile *************************
 *
 * Open a file called IterationInfo.<ElastixLevel>.R<Resolution>.txt,
 * which will contain the iteration info table.
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::OpenIterationInfoFile()
{
  /** Remove the current iteration info output file, if any. */
  this->GetIterationInfo().RemoveOutputFile();

  if (ElastixBase::m_IterationInfoFile.is_open())
  {
    ElastixBase::m_IterationInfoFile.close();
  }

  const Configuration & configuration = itk::Deref(ElastixBase::GetConfiguration());

  if (const std::string outputDirectoryPath = configuration.GetCommandLineArgument("-out");
      !outputDirectoryPath.empty())
  {
    /** Create the IterationInfo filename for this resolution. */
    std::ostringstream makeFileName;
    makeFileName << outputDirectoryPath << "IterationInfo." << configuration.GetElastixLevel() << ".R"
                 << this->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel() << ".txt";
    std::string fileName = makeFileName.str();

    /** Open the IterationInfoFile. */
    ElastixBase::m_IterationInfoFile.open(fileName);
    if (!(ElastixBase::m_IterationInfoFile.is_open()))
    {
      log::error(std::ostringstream{} << "ERROR: File \"" << fileName << "\" could not be opened!");
    }
    else
    {
      /** Add this file to the list of outputs of IterationInfo. */
      this->GetIterationInfo().SetOutputFile(ElastixBase::m_IterationInfoFile);
    }
  }

} // end OpenIterationInfoFile()


/**
 * ************** GetOriginalFixedImageDirection *********************
 * Determine the original fixed image direction (it might have been
 * set to identity by setting the UseDirectionCosines option to false).
 * This function retrieves the true direction cosines.
 */

template <typename TFixedImage, typename TMovingImage>
bool
ElastixTemplate<TFixedImage, TMovingImage>::GetOriginalFixedImageDirection(FixedImageDirectionType & direction) const
{
  if (this->GetFixedImage() == nullptr)
  {
    const Configuration & configuration = itk::Deref(ElastixBase::GetConfiguration());

    /** Try to read direction cosines from (transform-)parameter file. (The matrix elements must be specified in
     * column-major order.) */
    FixedImageDirectionType directionRead = direction;
    for (unsigned int i = 0; i < FixedDimension; ++i)
    {
      for (unsigned int j = 0; j < FixedDimension; ++j)
      {
        if (!configuration.ReadParameter(directionRead(j, i), "Direction", i * FixedDimension + j, false))
        {
          return false;
        }
      }
    }
    direction = directionRead;
    return true;
  }

  /** Only trust this when the fixed image exists. */
  if (ElastixBase::m_OriginalFixedImageDirectionFlat.size() == FixedDimension * FixedDimension)
  {
    for (unsigned int i = 0; i < FixedDimension; ++i)
    {
      for (unsigned int j = 0; j < FixedDimension; ++j)
      {
        direction(j, i) = ElastixBase::m_OriginalFixedImageDirectionFlat[i * FixedDimension + j];
      }
    }
    return true;
  }
  else
  {
    return false;
  }
} // end GetOriginalFixedImageDirection()


/**
 * ************** SetOriginalFixedImageDirection *********************
 */

template <typename TFixedImage, typename TMovingImage>
void
ElastixTemplate<TFixedImage, TMovingImage>::SetOriginalFixedImageDirection(const FixedImageDirectionType & arg)
{
  /** flatten to 1d array */
  ElastixBase::m_OriginalFixedImageDirectionFlat.resize(FixedDimension * FixedDimension);
  for (unsigned int i = 0; i < FixedDimension; ++i)
  {
    for (unsigned int j = 0; j < FixedDimension; ++j)
    {
      ElastixBase::m_OriginalFixedImageDirectionFlat[i * FixedDimension + j] = arg(j, i);
    }
  }

} // end SetOriginalFixedImageDirection()


} // end namespace elastix

#endif // end #ifndef elxElastixTemplate_hxx

#undef elxCheckAndSetComponentMacro
