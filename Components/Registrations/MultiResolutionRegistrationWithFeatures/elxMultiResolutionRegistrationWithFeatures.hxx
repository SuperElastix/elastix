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
#ifndef elxMultiResolutionRegistrationWithFeatures_hxx
#define elxMultiResolutionRegistrationWithFeatures_hxx

#include "elxMultiResolutionRegistrationWithFeatures.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
MultiResolutionRegistrationWithFeatures<TElastix>::BeforeRegistration()
{
  /** Get the components from this->m_Elastix and set them. */
  this->GetAndSetComponents();

  /** Set the number of resolutions. */
  unsigned int numberOfResolutions = 3;
  this->m_Configuration->ReadParameter(numberOfResolutions, "NumberOfResolutions", 0);
  this->SetNumberOfLevels(numberOfResolutions);

  /** Set the FixedImageRegions to the buffered regions. */
  this->GetAndSetFixedImageRegions();

  /** Set the fixed image interpolators. */
  this->GetAndSetFixedImageInterpolators();

} // end BeforeRegistration()


/**
 * ******************* BeforeEachResolution ***********************
 */

template <class TElastix>
void
MultiResolutionRegistrationWithFeatures<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = this->GetCurrentLevel();

  /** Set the masks in the metric. */
  this->UpdateFixedMasks(level);
  this->UpdateMovingMasks(level);

} // end BeforeEachResolution()


/**
 * *********************** GetAndSetComponents ************************
 */

template <class TElastix>
void
MultiResolutionRegistrationWithFeatures<TElastix>::GetAndSetComponents()
{
  /** Get the component from this->GetElastix() (as elx::...BaseType *),
   * cast it to the appropriate type and set it in 'this'.
   */

  /** Set the metric. */
  MetricType * testPtr = dynamic_cast<MetricType *>(this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType());
  if (testPtr)
  {
    this->SetMetric(testPtr);
  }
  else
  {
    itkExceptionMacro(
      << "ERROR: MultiResolutionRegistrationWithFeatures expects the metric to be of type AdvancedImageToImageMetric!");
  }

  /** Set the fixed images. */
  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImages(); ++i)
  {
    this->SetFixedImage(this->GetElastix()->GetFixedImage(i), i);
  }

  /** Set the moving images. */
  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfMovingImages(); ++i)
  {
    this->SetMovingImage(this->GetElastix()->GetMovingImage(i), i);
  }

  /** Set the fixed image pyramids. */
  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImagePyramids(); ++i)
  {
    this->SetFixedImagePyramid(this->GetElastix()->GetElxFixedImagePyramidBase(i)->GetAsITKBaseType(), i);
  }

  /** Set the moving image pyramids. */
  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfMovingImagePyramids(); ++i)
  {
    this->SetMovingImagePyramid(this->GetElastix()->GetElxMovingImagePyramidBase(i)->GetAsITKBaseType(), i);
  }

  /** Set the moving image interpolators. */
  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfInterpolators(); ++i)
  {
    this->SetInterpolator(this->GetElastix()->GetElxInterpolatorBase(i)->GetAsITKBaseType(), i);
  }

  /** Set the optimizer. */
  this->SetOptimizer(dynamic_cast<OptimizerType *>(this->GetElastix()->GetElxOptimizerBase()->GetAsITKBaseType()));

  /** Set the transform. */
  this->SetTransform(this->GetElastix()->GetElxTransformBase()->GetAsITKBaseType());

  /** Samplers are not always needed: */
  if (this->GetElastix()->GetElxMetricBase()->GetAdvancedMetricUseImageSampler())
  {
    if (this->GetElastix()->GetElxImageSamplerBase())
    {
      this->GetElastix()->GetElxMetricBase()->SetAdvancedMetricImageSampler(
        this->GetElastix()->GetElxImageSamplerBase()->GetAsITKBaseType());
    }
    else
    {
      xl::xout["error"] << "No ImageSampler has been specified." << std::endl;
      itkExceptionMacro(<< "The metric requires an ImageSampler, but it is not available!");
    }
  }

} // end GetAndSetComponents()


/**
 * *********************** GetAndSetFixedImageRegions ************************
 */

template <class TElastix>
void
MultiResolutionRegistrationWithFeatures<TElastix>::GetAndSetFixedImageRegions()
{
  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImages(); ++i)
  {
    /** Make sure the fixed image is up to date. */
    try
    {
      this->GetElastix()->GetFixedImage(i)->Update();
    }
    catch (itk::ExceptionObject & excp)
    {
      /** Add information to the exception. */
      excp.SetLocation("MultiResolutionRegistrationWithFeatures - BeforeRegistration()");
      std::string err_str = excp.GetDescription();
      err_str += "\nError occured while updating region info of the fixed image.\n";
      excp.SetDescription(err_str);
      /** Pass the exception to an higher level. */
      throw;
    }

    /** Set the fixed image region. */
    this->SetFixedImageRegion(this->GetElastix()->GetFixedImage(i)->GetBufferedRegion(), i);
  }

} // end GetAndSetFixedImageRegions()


/**
 * *********************** GetAndSetFixedImageInterpolators ************************
 */

template <class TElastix>
void
MultiResolutionRegistrationWithFeatures<TElastix>::GetAndSetFixedImageInterpolators()
{
  /** Shrot cut. */
  const unsigned int noFixIm = this->GetNumberOfFixedImages();

  /** Get the spline order of the fixed feature image interpolators. */
  unsigned int splineOrder = 1;
  this->m_Configuration->ReadParameter(splineOrder, "FixedImageInterpolatorBSplineOrder", 0);
  std::vector<unsigned int> soFII(noFixIm, splineOrder);
  for (unsigned int i = 1; i < noFixIm; ++i)
  {
    this->m_Configuration->ReadParameter(soFII[i], "FixedImageInterpolatorBSplineOrder", i, false);
  }

  /** Create and set interpolators for the fixed feature images. */
  using FixedImageInterpolatorType = itk::BSplineInterpolateImageFunction<FixedImageType>;
  using FixedImageInterpolatorVectorType = std::vector<typename FixedImageInterpolatorType::Pointer>;
  FixedImageInterpolatorVectorType interpolators(noFixIm);
  for (unsigned int i = 0; i < noFixIm; ++i)
  {
    interpolators[i] = FixedImageInterpolatorType::New();
    interpolators[i]->SetSplineOrder(soFII[i]);
    this->SetFixedImageInterpolator(interpolators[i], i);
  }

} // end GetAndSetFixedImageInterpolators()


/**
 * ************************* UpdateFixedMasks ************************
 */

template <class TElastix>
void
MultiResolutionRegistrationWithFeatures<TElastix>::UpdateFixedMasks(unsigned int level)
{
  /** Use only one mask. */
  const unsigned int nrOfFixedImageMasks = 1; // this->GetElastix()->GetNumberOfFixedMasks();

  /** Array of bools, that remembers for each mask if erosion is wanted. */
  UseMaskErosionArrayType useMaskErosionArray;

  /** Bool that remembers if mask erosion is wanted in any of the masks
   * remains false when no masks are used.
   */
  bool useMaskErosion;

  /** Read whether mask erosion is wanted, if any masks were supplied. */
  useMaskErosion = this->ReadMaskParameters(useMaskErosionArray, nrOfFixedImageMasks, "Fixed", level);

  /** Create and start timer, to time the whole mask configuration procedure. */
  itk::TimeProbe timer;
  timer.Start();

  /** Set the fixed image mask. Only one mask is assumed here. */
  FixedMaskSpatialObjectPointer fixedMask = this->GenerateFixedMaskSpatialObject(
    this->GetElastix()->GetFixedMask(), useMaskErosion, this->GetFixedImagePyramid(), level);
  this->GetModifiableMultiInputMetric()->SetFixedImageMask(fixedMask);

  /** Stop timer and print the elapsed time. */
  timer.Stop();
  elxout << "Setting the fixed masks took: " << static_cast<long>(timer.GetMean() * 1000) << " ms." << std::endl;

} // end UpdateFixedMasks()


/**
 * ************************* UpdateMovingMasks ************************
 */

template <class TElastix>
void
MultiResolutionRegistrationWithFeatures<TElastix>::UpdateMovingMasks(unsigned int level)
{
  /** Use only one mask. */
  const unsigned int nrOfMovingImageMasks = 1; // this->GetElastix()->GetNumberOfMovingMasks();

  /** Array of bools, that remembers for each mask if erosion is wanted. */
  UseMaskErosionArrayType useMaskErosionArray;

  /** Bool that remembers if mask erosion is wanted in any of the masks
   * remains false when no masks are used.
   */
  bool useMaskErosion;

  /** Read whether mask erosion is wanted, if any masks were supplied. */
  useMaskErosion = this->ReadMaskParameters(useMaskErosionArray, nrOfMovingImageMasks, "Moving", level);

  /** Create and start timer, to time the whole mask configuration procedure. */
  itk::TimeProbe timer;
  timer.Start();

  /** Set the moving image mask. Only one mask is assumed here. */
  MovingMaskSpatialObjectPointer movingMask = this->GenerateMovingMaskSpatialObject(
    this->GetElastix()->GetMovingMask(), useMaskErosion, this->GetMovingImagePyramid(), level);
  this->GetModifiableMultiInputMetric()->SetMovingImageMask(movingMask);

  /** Stop timer and print the elapsed time. */
  timer.Stop();
  elxout << "Setting the moving masks took: " << static_cast<long>(timer.GetMean() * 1000) << " ms." << std::endl;

} // end UpdateMovingMasks()


} // end namespace elastix

#endif // end #ifndef elxMultiResolutionRegistrationWithFeatures_hxx
