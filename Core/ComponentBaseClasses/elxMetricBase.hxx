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
#ifndef elxMetricBase_hxx
#define elxMetricBase_hxx

#include "elxMetricBase.h"

namespace elastix
{

/**
 * ******************* BeforeEachResolutionBase ******************
 */

template <class TElastix>
void
MetricBase<TElastix>::BeforeEachResolutionBase()
{
  /** Get the current resolution level. */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Check if the exact metric value, computed on all pixels, should be shown. */

  /** Define the name of the ExactMetric column */
  std::string exactMetricColumn = "Exact";
  exactMetricColumn += this->GetComponentLabel();

  /** Remove the ExactMetric-column, if it already existed. */
  this->RemoveTargetCellFromIterationInfo(exactMetricColumn.c_str());

  /** Read the parameter file: Show the exact metric in every iteration? */
  bool showExactMetricValue = false;
  this->GetConfiguration()->ReadParameter(
    showExactMetricValue, "ShowExactMetricValue", this->GetComponentLabel(), level, 0);
  this->m_ShowExactMetricValue = showExactMetricValue;
  if (showExactMetricValue)
  {
    /** Create a new column in the iteration info table */
    this->AddTargetCellToIterationInfo(exactMetricColumn.c_str());
    this->GetIterationInfoAt(exactMetricColumn.c_str()) << std::showpoint << std::fixed;
  }

  /** Read the sample grid spacing for computing the "exact" metric */
  if (showExactMetricValue)
  {
    using SampleGridSpacingValueType = typename ExactMetricImageSamplerType::SampleGridSpacingValueType;
    this->m_ExactMetricSampleGridSpacing.Fill(1);

    /** Read the desired grid spacing of the samples. */
    unsigned int spacing_dim;
    for (unsigned int dim = 0; dim < FixedImageDimension; ++dim)
    {
      spacing_dim = this->m_ExactMetricSampleGridSpacing[dim];
      this->GetConfiguration()->ReadParameter(
        spacing_dim, "ExactMetricSampleGridSpacing", this->GetComponentLabel(), level * FixedImageDimension + dim, -1);
      this->m_ExactMetricSampleGridSpacing[dim] = static_cast<SampleGridSpacingValueType>(spacing_dim);
    }

    /** Read the requested frequency of exact metric evaluation. */
    unsigned int eachXNumberOfIterations = 1;
    this->GetConfiguration()->ReadParameter(
      eachXNumberOfIterations, "ExactMetricEveryXIterations", this->GetComponentLabel(), level, 0);
    this->m_ExactMetricEachXNumberOfIterations = eachXNumberOfIterations;
  }

  /** Cast this to AdvancedMetricType. */
  AdvancedMetricType * thisAsAdvanced = dynamic_cast<AdvancedMetricType *>(this);

  /** For advanced metrics several other things can be set. */
  if (thisAsAdvanced != nullptr)
  {
    /** Should the metric check for enough samples? */
    bool checkNumberOfSamples = true;
    this->GetConfiguration()->ReadParameter(
      checkNumberOfSamples, "CheckNumberOfSamples", this->GetComponentLabel(), level, 0);

    /** Get the required ratio. */
    float ratio = 0.25;
    this->GetConfiguration()->ReadParameter(
      ratio, "RequiredRatioOfValidSamples", this->GetComponentLabel(), level, 0, false);

    /** Set it. */
    if (!checkNumberOfSamples)
    {
      thisAsAdvanced->SetRequiredRatioOfValidSamples(0.0);
    }
    else
    {
      thisAsAdvanced->SetRequiredRatioOfValidSamples(ratio);
    }

    /** Set moving image derivative scales. */
    std::size_t usescales = this->GetConfiguration()->CountNumberOfParameterEntries("MovingImageDerivativeScales");
    if (usescales == 0)
    {
      thisAsAdvanced->SetUseMovingImageDerivativeScales(false);
      thisAsAdvanced->SetScaleGradientWithRespectToMovingImageOrientation(false);
    }
    else
    {
      thisAsAdvanced->SetUseMovingImageDerivativeScales(true);

      /** Read the scales from the parameter file. */
      MovingImageDerivativeScalesType movingImageDerivativeScales;
      movingImageDerivativeScales.Fill(1.0);
      for (unsigned int i = 0; i < MovingImageDimension; ++i)
      {
        this->GetConfiguration()->ReadParameter(
          movingImageDerivativeScales[i], "MovingImageDerivativeScales", this->GetComponentLabel(), i, -1, false);
      }

      /** Set and report. */
      thisAsAdvanced->SetMovingImageDerivativeScales(movingImageDerivativeScales);
      elxout << "Multiplying moving image derivatives by: " << movingImageDerivativeScales << std::endl;

      /** Check if the scales are applied taking into account the moving image orientation. */
      bool wrtMoving = false;
      this->GetConfiguration()->ReadParameter(
        wrtMoving, "ScaleGradientWithRespectToMovingImageOrientation", this->GetComponentLabel(), level, false);
      thisAsAdvanced->SetScaleGradientWithRespectToMovingImageOrientation(wrtMoving);
    }

    /** Should the metric use multi-threading? */
    bool useMultiThreading = true;
    this->GetConfiguration()->ReadParameter(
      useMultiThreading, "UseMultiThreadingForMetrics", this->GetComponentLabel(), level, 0);

    thisAsAdvanced->SetUseMultiThread(useMultiThreading);
    if (useMultiThreading)
    {
      std::string tmp = this->m_Configuration->GetCommandLineArgument("-threads");
      if (!tmp.empty())
      {
        const unsigned int nrOfThreads = atoi(tmp.c_str());
        thisAsAdvanced->SetNumberOfWorkUnits(nrOfThreads);
      }
    }

  } // end advanced metric

} // end BeforeEachResolutionBase()


/**
 * ******************* AfterEachIterationBase ******************
 */

template <class TElastix>
void
MetricBase<TElastix>::AfterEachIterationBase()
{
  /** Show the metric value computed on all voxels, if the user wanted it. */

  /** Define the name of the ExactMetric column (ExactMetric<i>). */
  std::string exactMetricColumn = "Exact";
  exactMetricColumn += this->GetComponentLabel();

  this->m_CurrentExactMetricValue = 0.0;
  if (this->m_ShowExactMetricValue &&
      (this->m_Elastix->GetIterationCounter() % this->m_ExactMetricEachXNumberOfIterations == 0))
  {
    this->m_CurrentExactMetricValue =
      this->GetExactValue(this->GetElastix()->GetElxOptimizerBase()->GetAsITKBaseType()->GetCurrentPosition());

    this->GetIterationInfoAt(exactMetricColumn.c_str()) << this->m_CurrentExactMetricValue;
  }

} // end AfterEachIterationBase()


/**
 * ********************* SelectNewSamples ************************
 */

template <class TElastix>
void
MetricBase<TElastix>::SelectNewSamples()
{
  if (this->GetAdvancedMetricImageSampler())
  {
    /** Force the metric to base its computation on a new subset of image samples. */
    this->GetAdvancedMetricImageSampler()->SelectNewSamplesOnUpdate();
  }
  else
  {
    /** Not every metric may have implemented this, so give a warning when this
     * method is called for a metric without sampler support.
     * To avoid the warning, this method may be overridden by a subclass.
     */
    xl::xout["warning"] << "WARNING: The NewSamplesEveryIteration option was set to \"true\", but "
                        << this->GetComponentLabel() << " does not use a sampler." << std::endl;
  }

} // end SelectNewSamples()


/**
 * ********************* GetExactValue ************************
 */

template <class TElastix>
auto
MetricBase<TElastix>::GetExactValue(const ParametersType & parameters) -> MeasureType
{
  /** Get the current image sampler. */
  typename ImageSamplerBaseType::Pointer currentSampler = this->GetAdvancedMetricImageSampler();

  /** Useless implementation if no image sampler is used; we may as
   * well throw an error, but the ShowExactMetricValue is not really
   * essential for good registration...
   */
  if (currentSampler.IsNull())
  {
    return itk::NumericTraits<MeasureType>::Zero;
  }

  /** Try to cast the current Sampler to a FullSampler. */
  ExactMetricImageSamplerType * testPointer = dynamic_cast<ExactMetricImageSamplerType *>(currentSampler.GetPointer());
  if (testPointer != nullptr)
  {
    /** GetValue gives us the exact value! */
    return this->GetAsITKBaseType()->GetValue(parameters);
  }

  /** We have to provide the metric a full (or actually 'grid') sampler,
   * calls its GetValue and set back its original sampler.
   */
  if (this->m_ExactMetricSampler.IsNull())
  {
    this->m_ExactMetricSampler = ExactMetricImageSamplerType::New();
  }

  /** Copy settings from current sampler. */
  this->m_ExactMetricSampler->SetInput(currentSampler->GetInput());
  this->m_ExactMetricSampler->SetMask(currentSampler->GetMask());
  this->m_ExactMetricSampler->SetInputImageRegion(currentSampler->GetInputImageRegion());
  this->m_ExactMetricSampler->SetSampleGridSpacing(this->m_ExactMetricSampleGridSpacing);
  this->m_ExactMetricSampler->Update();
  this->SetAdvancedMetricImageSampler(this->m_ExactMetricSampler);

  /** Compute the metric value on the full images. */
  MeasureType exactValue = this->GetAsITKBaseType()->GetValue(parameters);

  /** Reset the original sampler. */
  this->SetAdvancedMetricImageSampler(currentSampler);

  return exactValue;

} // end GetExactValue()


/**
 * ******************* GetAdvancedMetricUseImageSampler ********************
 */

template <class TElastix>
bool
MetricBase<TElastix>::GetAdvancedMetricUseImageSampler() const
{
  /** Cast this to AdvancedMetricType. */
  const AdvancedMetricType * thisAsMetricWithSampler = dynamic_cast<const AdvancedMetricType *>(this);

  /** If no AdvancedMetricType, return false. */
  if (thisAsMetricWithSampler == nullptr)
  {
    return false;
  }

  return thisAsMetricWithSampler->GetUseImageSampler();

} // end GetAdvancedMetricUseImageSampler()


/**
 * ******************* SetAdvancedMetricImageSampler ********************
 */

template <class TElastix>
void
MetricBase<TElastix>::SetAdvancedMetricImageSampler(ImageSamplerBaseType * sampler)
{
  /** Cast this to AdvancedMetricType. */
  AdvancedMetricType * thisAsMetricWithSampler = dynamic_cast<AdvancedMetricType *>(this);

  /** If no AdvancedMetricType, or if the MetricWithSampler does not
   * utilize the sampler, return.
   */
  if (thisAsMetricWithSampler == nullptr)
  {
    return;
  }
  if (thisAsMetricWithSampler->GetUseImageSampler() == false)
  {
    return;
  }

  /** Set the sampler. */
  thisAsMetricWithSampler->SetImageSampler(sampler);

} // end SetAdvancedMetricImageSampler()


/**
 * ******************* GetAdvancedMetricImageSampler ********************
 */

template <class TElastix>
auto
MetricBase<TElastix>::GetAdvancedMetricImageSampler() const -> ImageSamplerBaseType *
{
  /** Cast this to AdvancedMetricType. */
  const AdvancedMetricType * thisAsMetricWithSampler = dynamic_cast<const AdvancedMetricType *>(this);

  /** If no AdvancedMetricType, or if the MetricWithSampler does not
   * utilize the sampler, return 0.
   */
  if (thisAsMetricWithSampler == nullptr)
  {
    return nullptr;
  }
  if (thisAsMetricWithSampler->GetUseImageSampler() == false)
  {
    return nullptr;
  }

  /** Return the sampler. */
  return thisAsMetricWithSampler->GetImageSampler();

} // end GetAdvancedMetricImageSampler()

} // end namespace elastix

#endif // end #ifndef elxMetricBase_hxx
