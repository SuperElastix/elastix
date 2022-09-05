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
#ifndef elxResamplerBase_hxx
#define elxResamplerBase_hxx

#include "elxResamplerBase.h"
#include "elxConversion.h"

#include "itkImageFileCastWriter.h"
#include "itkChangeInformationImageFilter.h"
#include "itkAdvancedRayCastInterpolateImageFunction.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* BeforeRegistrationBase *******************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::BeforeRegistrationBase()
{
  /** Connect the components. */
  this->SetComponents();

  /** Set the size of the image to be produced by the resampler. */

  /** Get a pointer to the fixedImage.
   * \todo make it a cast to the fixed image type
   */
  using FixedImageType = typename ElastixType::FixedImageType;
  FixedImageType * fixedImage = this->m_Elastix->GetFixedImage();
  ITKBaseType &    resampleImageFilter = this->GetSelf();

  /** Set the region info to the same values as in the fixedImage. */
  resampleImageFilter.SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
  resampleImageFilter.SetOutputStartIndex(fixedImage->GetLargestPossibleRegion().GetIndex());
  resampleImageFilter.SetOutputOrigin(fixedImage->GetOrigin());
  resampleImageFilter.SetOutputSpacing(fixedImage->GetSpacing());
  resampleImageFilter.SetOutputDirection(fixedImage->GetDirection());

  /** Set the DefaultPixelValue (for pixels in the resampled image
   * that come from outside the original (moving) image.
   */
  OutputPixelType defaultPixelValue{};
  this->m_Configuration->ReadParameter(defaultPixelValue, "DefaultPixelValue", 0, false);

  /** Set the defaultPixelValue. */
  resampleImageFilter.SetDefaultPixelValue(defaultPixelValue);

} // end BeforeRegistrationBase()


/**
 * ******************* AfterEachResolutionBase ********************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::AfterEachResolutionBase()
{
  /** Set the final transform parameters. */
  this->GetElastix()->GetElxTransformBase()->SetFinalParameters();

  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Decide whether or not to write the result image this resolution. */
  bool writeResultImageThisResolution = false;
  this->m_Configuration->ReadParameter(
    writeResultImageThisResolution, "WriteResultImageAfterEachResolution", "", level, 0, false);

  /** Writing result image. */
  if (writeResultImageThisResolution)
  {
    /** Create a name for the final result. */
    std::string resultImageFormat = "mhd";
    this->m_Configuration->ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);
    std::ostringstream makeFileName;
    makeFileName << this->m_Configuration->GetCommandLineArgument("-out") << "result."
                 << this->m_Configuration->GetElastixLevel() << ".R" << level << "." << resultImageFormat;

    /** Time the resampling. */
    itk::TimeProbe timer;
    timer.Start();

    /** Apply the final transform, and save the result. */
    elxout << "Applying transform this resolution ..." << std::endl;
    try
    {
      this->ResampleAndWriteResultImage(makeFileName.str().c_str());
    }
    catch (const itk::ExceptionObject & excp)
    {
      xl::xout["error"] << "Exception caught: " << std::endl;
      xl::xout["error"] << excp << "Resuming elastix." << std::endl;
    }

    /** Print the elapsed time for the resampling. */
    timer.Stop();
    elxout << "  Applying transform took " << Conversion::SecondsToDHMS(timer.GetMean(), 2) << std::endl;

  } // end if

} // end AfterEachResolutionBase()


/**
 * ******************* AfterEachIterationBase ********************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::AfterEachIterationBase()
{
  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** What is the current iteration number? */
  const unsigned int iter = this->m_Elastix->GetIterationCounter();

  /** Decide whether or not to write the result image this iteration. */
  bool writeResultImageThisIteration = false;
  this->m_Configuration->ReadParameter(
    writeResultImageThisIteration, "WriteResultImageAfterEachIteration", "", level, 0, false);

  /** Writing result image. */
  if (writeResultImageThisIteration)
  {
    /** Set the final transform parameters. */
    this->GetElastix()->GetElxTransformBase()->SetFinalParameters();

    /** Create a name for the final result. */
    std::string resultImageFormat = "mhd";
    this->m_Configuration->ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);
    std::ostringstream makeFileName;
    makeFileName << this->m_Configuration->GetCommandLineArgument("-out") << "result."
                 << this->m_Configuration->GetElastixLevel() << ".R" << level << ".It" << std::setfill('0')
                 << std::setw(7) << iter << "." << resultImageFormat;

    /** Apply the final transform, and save the result. */
    try
    {
      this->ResampleAndWriteResultImage(makeFileName.str().c_str(), false);
    }
    catch (const itk::ExceptionObject & excp)
    {
      xl::xout["error"] << "Exception caught: " << std::endl;
      xl::xout["error"] << excp << "Resuming elastix." << std::endl;
    }

  } // end if

} // end AfterEachIterationBase()


/**
 * ******************* AfterRegistrationBase ********************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::AfterRegistrationBase()
{
  /** Set the final transform parameters. */
  this->GetElastix()->GetElxTransformBase()->SetFinalParameters();

  /** Decide whether or not to write the result image. */
  std::string writeResultImage = "true";
  this->m_Configuration->ReadParameter(writeResultImage, "WriteResultImage", 0);

  const auto isElastixLibrary = BaseComponent::IsElastixLibrary();

  /** The library interface may executed multiple times in
   * a session in which case the images should not be released
   * However, if this is not the library interface:
   * Release memory to be able to resample in case a limited
   * amount of memory is available.
   */
  bool releaseMemoryBeforeResampling{ !isElastixLibrary };

  this->m_Configuration->ReadParameter(releaseMemoryBeforeResampling, "ReleaseMemoryBeforeResampling", 0, false);
  if (releaseMemoryBeforeResampling)
  {
    this->ReleaseMemory();
  }

  /**
   * Create the result image and put it in ResultImageContainer
   * Only necessary when compiling elastix as a library!
   */
  if (isElastixLibrary)
  {
    if (writeResultImage == "true")
    {
      this->CreateItkResultImage();
    }
  }
  else
  {
    /** Writing result image. */
    if (writeResultImage == "true")
    {
      /** Create a name for the final result. */
      std::string resultImageFormat = "mhd";
      this->m_Configuration->ReadParameter(resultImageFormat, "ResultImageFormat", 0);
      std::ostringstream makeFileName;
      makeFileName << this->m_Configuration->GetCommandLineArgument("-out") << "result."
                   << this->m_Configuration->GetElastixLevel() << "." << resultImageFormat;

      /** Time the resampling. */
      itk::TimeProbe timer;
      timer.Start();

      /** Apply the final transform, and save the result,
       * by calling ResampleAndWriteResultImage.
       */
      elxout << "\nApplying final transform ..." << std::endl;
      try
      {
        this->ResampleAndWriteResultImage(makeFileName.str().c_str(), this->m_ShowProgress);
      }
      catch (const itk::ExceptionObject & excp)
      {
        xl::xout["error"] << "Exception caught: " << std::endl;
        xl::xout["error"] << excp << "Resuming elastix." << std::endl;
      }

      /** Print the elapsed time for the resampling. */
      timer.Stop();
      elxout << "  Applying final transform took " << Conversion::SecondsToDHMS(timer.GetMean(), 2) << std::endl;
    }
    else
    {
      /** Do not apply the final transform. */
      elxout << '\n' << "Skipping applying final transform, no resulting output image generated." << std::endl;
    } // end if
  }

} // end AfterRegistrationBase()


/**
 * *********************** SetComponents ************************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::SetComponents()
{
  /** Set the transform, the interpolator and the inputImage
   * (which is the moving image).
   */
  ITKBaseType & resampleImageFilter = this->GetSelf();

  resampleImageFilter.SetTransform(BaseComponent::AsITKBaseType(this->m_Elastix->GetElxTransformBase()));

  resampleImageFilter.SetInterpolator(BaseComponent::AsITKBaseType(this->m_Elastix->GetElxResampleInterpolatorBase()));

  resampleImageFilter.SetInput(this->m_Elastix->GetMovingImage());

} // end SetComponents()


/**
 * ******************* ResampleAndWriteResultImage ********************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::ResampleAndWriteResultImage(const char * filename, const bool showProgress)
{
  ITKBaseType & resampleImageFilter = this->GetSelf();

  /** Make sure the resampler is updated. */
  resampleImageFilter.Modified();

  /** Add a progress observer to the resampler. */
  const auto progressObserver = BaseComponent::IsElastixLibrary() ? nullptr : ProgressCommandType::New();
  if (showProgress && (progressObserver != nullptr))
  {
    progressObserver->ConnectObserver(this->GetAsITKBaseType());
    progressObserver->SetStartString("  Progress: ");
    progressObserver->SetEndString("%");
  }

  /** Do the resampling. */
  try
  {
    resampleImageFilter.Update();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("ResamplerBase - WriteResultImage()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while resampling the image.\n";
    excp.SetDescription(err_str);

    /** Pass the exception to an higher level. */
    throw;
  }

  /** Perform the writing. */
  this->WriteResultImage(resampleImageFilter.GetOutput(), filename, showProgress);

  /** Disconnect from the resampler. */
  if (showProgress && (progressObserver != nullptr))
  {
    progressObserver->DisconnectObserver(this->GetAsITKBaseType());
  }

} // end ResampleAndWriteResultImage()


/**
 * ******************* WriteResultImage ********************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::WriteResultImage(OutputImageType * image, const char * filename, const bool showProgress)
{
  ITKBaseType & resampleImageFilter = this->GetSelf();

  /** Check if ResampleInterpolator is the RayCastResampleInterpolator  */
  const auto testptr = dynamic_cast<itk::AdvancedRayCastInterpolateImageFunction<InputImageType, CoordRepType> *>(
    resampleImageFilter.GetInterpolator());

  /** If RayCastResampleInterpolator is used reset the Transform to
   * overrule default Resampler settings.
   */

  if (testptr != nullptr)
  {
    resampleImageFilter.SetTransform(testptr->GetTransform());
  }

  /** Read output pixeltype from parameter the file. Replace possible " " with "_". */
  std::string resultImagePixelType = "short";
  this->m_Configuration->ReadParameter(resultImagePixelType, "ResultImagePixelType", 0, false);
  const std::string::size_type pos = resultImagePixelType.find(" ");
  if (pos != std::string::npos)
  {
    resultImagePixelType.replace(pos, 1, "_");
  }

  /** Read from the parameter file if compression is desired. */
  bool doCompression = false;
  this->m_Configuration->ReadParameter(doCompression, "CompressResultImage", 0, false);

  /** Typedef's for writing the output image. */
  using ChangeInfoFilterType = itk::ChangeInformationImageFilter<OutputImageType>;

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false.
   */
  auto          infoChanger = ChangeInfoFilterType::New();
  DirectionType originalDirection;
  bool          retdc = this->GetElastix()->GetOriginalFixedImageDirection(originalDirection);
  infoChanger->SetOutputDirection(originalDirection);
  infoChanger->SetChangeDirection(retdc & !this->GetElastix()->GetUseDirectionCosines());
  infoChanger->SetInput(image);

  /** Do the writing. */
  if (showProgress)
  {
    xl::xout["coutonly"] << "\n  Writing image ..." << std::endl;
  }
  try
  {
    itk::WriteCastedImage(*(infoChanger->GetOutput()), filename, resultImagePixelType, doCompression);
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("ResamplerBase - AfterRegistrationBase()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing resampled image.\n";
    excp.SetDescription(err_str);

    /** Pass the exception to an higher level. */
    throw;
  }
} // end WriteResultImage()


/*
 * ******************* CreateItkResultImage ********************
 * \todo: avoid code duplication with WriteResultImage function
 */

template <class TElastix>
void
ResamplerBase<TElastix>::CreateItkResultImage()
{
  itk::DataObject::Pointer resultImage;
  ITKBaseType &            resampleImageFilter = this->GetSelf();

  /** Make sure the resampler is updated. */
  resampleImageFilter.Modified();

  const auto progressObserver =
    BaseComponent::IsElastixLibrary() ? nullptr : ProgressCommandType::CreateAndConnect(*(this->GetAsITKBaseType()));

  /** Do the resampling. */
  try
  {
    resampleImageFilter.Update();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("ResamplerBase - WriteResultImage()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while resampling the image.\n";
    excp.SetDescription(err_str);

    /** Pass the exception to an higher level. */
    throw;
  }

  /** Check if ResampleInterpolator is the RayCastResampleInterpolator */
  const auto testptr = dynamic_cast<itk::AdvancedRayCastInterpolateImageFunction<InputImageType, CoordRepType> *>(
    resampleImageFilter.GetInterpolator());

  /** If RayCastResampleInterpolator is used reset the Transform to
   * overrule default Resampler settings */

  if (testptr != nullptr)
  {
    resampleImageFilter.SetTransform(testptr->GetTransform());
  }

  /** Read output pixeltype from parameter the file. */
  std::string resultImagePixelType = "short";
  this->m_Configuration->ReadParameter(resultImagePixelType, "ResultImagePixelType", 0, false);

  /** Typedef's for writing the output image. */
  using ChangeInfoFilterType = itk::ChangeInformationImageFilter<OutputImageType>;

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false.
   */
  auto          infoChanger = ChangeInfoFilterType::New();
  DirectionType originalDirection;
  bool          retdc = this->GetElastix()->GetOriginalFixedImageDirection(originalDirection);
  infoChanger->SetOutputDirection(originalDirection);
  infoChanger->SetChangeDirection(retdc & !this->GetElastix()->GetUseDirectionCosines());
  infoChanger->SetInput(resampleImageFilter.GetOutput());

  /** cast the image to the correct output image Type */
  if (resultImagePixelType == "char")
  {
    resultImage = CastImage<char>(infoChanger->GetOutput());
  }
  if (resultImagePixelType == "unsigned char")
  {
    resultImage = CastImage<unsigned char>(infoChanger->GetOutput());
  }
  else if (resultImagePixelType == "short")
  {
    resultImage = CastImage<short>(infoChanger->GetOutput());
  }
  else if (resultImagePixelType == "ushort" ||
           resultImagePixelType == "unsigned short") // <-- ushort for backwards compatibility
  {
    resultImage = CastImage<unsigned short>(infoChanger->GetOutput());
  }
  else if (resultImagePixelType == "int")
  {
    resultImage = CastImage<int>(infoChanger->GetOutput());
  }
  else if (resultImagePixelType == "unsigned int")
  {
    resultImage = CastImage<unsigned int>(infoChanger->GetOutput());
  }
  else if (resultImagePixelType == "long")
  {
    resultImage = CastImage<long>(infoChanger->GetOutput());
  }
  else if (resultImagePixelType == "unsigned long")
  {
    resultImage = CastImage<unsigned long>(infoChanger->GetOutput());
  }
  else if (resultImagePixelType == "float")
  {
    resultImage = CastImage<float>(infoChanger->GetOutput());
  }
  else if (resultImagePixelType == "double")
  {
    resultImage = CastImage<double>(infoChanger->GetOutput());
  }

  if (resultImage.IsNull())
  {
    itkExceptionMacro(<< "Unable to cast result image: ResultImagePixelType must be one of \"char\", \"unsigned "
                         "char\", \"short\", \"ushort\", \"unsigned short\", \"int\", \"unsigned int\", \"long\", "
                         "\"unsigned long\", \"float\" or \"double\" but was \""
                      << resultImagePixelType << "\".");
  }

  // put image in container
  this->m_Elastix->SetResultImage(resultImage);

  if (progressObserver != nullptr)
  {
    /** Disconnect from the resampler. */
    progressObserver->DisconnectObserver(this->GetAsITKBaseType());
  }
} // end CreateItkResultImage()


/*
 * ************************* ReadFromFile ***********************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::ReadFromFile()
{
  /** Connect the components. */
  this->SetComponents();

  /** Get spacing, origin and size of the image to be produced by the resampler. */
  SpacingType     spacing;
  IndexType       index;
  OriginPointType origin;
  SizeType        size = { { 0 } };
  auto            direction = DirectionType::GetIdentity();
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    /** No default size. Read size from the parameter file. */
    this->m_Configuration->ReadParameter(size[i], "Size", i);

    /** Default index. Read index from the parameter file. */
    index[i] = 0;
    this->m_Configuration->ReadParameter(index[i], "Index", i);

    /** Default spacing. Read spacing from the parameter file. */
    spacing[i] = 1.0;
    this->m_Configuration->ReadParameter(spacing[i], "Spacing", i);

    /** Default origin. Read origin from the parameter file. */
    origin[i] = 0.0;
    this->m_Configuration->ReadParameter(origin[i], "Origin", i);

    /** Read direction cosines. Default identity */
    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      this->m_Configuration->ReadParameter(direction(j, i), "Direction", i * ImageDimension + j);
    }
  }

  /** Check for image size. */
  unsigned int sum = 0;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    if (size[i] == 0)
    {
      ++sum;
    }
  }
  if (sum > 0)
  {
    xl::xout["error"] << "ERROR: One or more image sizes are 0 or unspecified!" << std::endl;
    /** \todo quit program nicely. */
  }

  ITKBaseType & resampleImageFilter = this->GetSelf();

  /** Set the region info to the same values as in the fixedImage. */
  resampleImageFilter.SetSize(size);
  resampleImageFilter.SetOutputStartIndex(index);
  resampleImageFilter.SetOutputOrigin(origin);
  resampleImageFilter.SetOutputSpacing(spacing);

  /** Set the direction cosines. If no direction cosines
   * should be used, set identity cosines, to simulate the
   * old ITK behavior.
   */
  if (!this->GetElastix()->GetUseDirectionCosines())
  {
    direction.SetIdentity();
  }
  resampleImageFilter.SetOutputDirection(direction);

  /** Set the DefaultPixelValue (for pixels in the resampled image
   * that come from outside the original (moving) image.
   */
  double defaultPixelValue = 0.0;
  bool   found = this->m_Configuration->ReadParameter(defaultPixelValue, "DefaultPixelValue", 0, false);

  if (found)
  {
    resampleImageFilter.SetDefaultPixelValue(static_cast<OutputPixelType>(defaultPixelValue));
  }

} // end ReadFromFile()


/**
 * ******************* WriteToFile ******************************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::WriteToFile(xl::xoutsimple & transformationParameterInfo) const
{
  ParameterMapType parameterMap;
  Self::CreateTransformParametersMap(parameterMap);

  /** Write resampler specific things. */
  transformationParameterInfo << ("\n// Resampler specific\n" + Conversion::ParameterMapToString(parameterMap));

} // end WriteToFile()


/**
 * ******************* CreateTransformParametersMap ******************************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::CreateTransformParametersMap(ParameterMapType & parameterMap) const
{
  /** Store the name of this transform. */
  parameterMap["Resampler"] = { this->elxGetClassName() };

  /** Store the DefaultPixelValue. */
  parameterMap["DefaultPixelValue"] = { Conversion::ToString(this->GetSelf().GetDefaultPixelValue()) };

  /** Store the output image format. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);
  parameterMap["ResultImageFormat"] = { resultImageFormat };

  /** Store output pixel type. */
  std::string resultImagePixelType = "short";
  this->m_Configuration->ReadParameter(resultImagePixelType, "ResultImagePixelType", 0, false);
  parameterMap["ResultImagePixelType"] = { resultImagePixelType };

  /** Store compression flag. */
  std::string doCompression = "false";
  this->m_Configuration->ReadParameter(doCompression, "CompressResultImage", 0, false);
  parameterMap["CompressResultImage"] = { doCompression };

  // Derived classes may add some extra parameters
  for (auto & keyAndValue : this->CreateDerivedTransformParametersMap())
  {
    const auto & key = keyAndValue.first;
    assert(parameterMap.count(key) == 0);
    parameterMap[key] = std::move(keyAndValue.second);
  }

} // end CreateTransformParametersMap()


/**
 * ******************* ReleaseMemory ********************
 */

template <class TElastix>
void
ResamplerBase<TElastix>::ReleaseMemory()
{
  /** Release some memory. Sometimes it is not possible to
   * resample and write an image, because too much memory is consumed by
   * elastix. Releasing some memory at this point helps a lot.
   */

  /** Release more memory, but only if this is the final elastix level. */
  if (this->GetConfiguration()->GetElastixLevel() + 1 == this->GetConfiguration()->GetTotalNumberOfElastixLevels())
  {
    /** Release fixed image memory. */
    const unsigned int nofi = this->GetElastix()->GetNumberOfFixedImages();
    for (unsigned int i = 0; i < nofi; ++i)
    {
      this->GetElastix()->GetFixedImage(i)->ReleaseData();
    }

    /** Release fixed mask image memory. */
    const unsigned int nofm = this->GetElastix()->GetNumberOfFixedMasks();
    for (unsigned int i = 0; i < nofm; ++i)
    {
      if (this->GetElastix()->GetFixedMask(i) != nullptr)
      {
        this->GetElastix()->GetFixedMask(i)->ReleaseData();
      }
    }

    /** Release moving mask image memory. */
    const unsigned int nomm = this->GetElastix()->GetNumberOfMovingMasks();
    for (unsigned int i = 0; i < nomm; ++i)
    {
      if (this->GetElastix()->GetMovingMask(i) != nullptr)
      {
        this->GetElastix()->GetMovingMask(i)->ReleaseData();
      }
    }

  } // end if final elastix level

  /** The B-spline interpolator stores a coefficient image of doubles the
   * size of the moving image. We clear it by setting the input image to
   * zero. The interpolator is not needed anymore, since we have the
   * resampler interpolator.
   */
  this->GetElastix()->GetElxInterpolatorBase()->GetAsITKBaseType()->SetInputImage(nullptr);

  // Clear ImageSampler, metric, optimizer, interpolator, registration, internal images?

} // end ReleaseMemory()


} // end namespace elastix

#endif // end #ifndef elxResamplerBase_hxx
