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
#ifndef elxMultiMetricMultiResolutionRegistration_hxx
#define elxMultiMetricMultiResolutionRegistration_hxx

#include "elxMultiMetricMultiResolutionRegistration.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* Constructor ***********************
 */

template <class TElastix>
MultiMetricMultiResolutionRegistration<TElastix>::MultiMetricMultiResolutionRegistration()
{
  this->m_ShowExactMetricValue = false;
} // end constructor


/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
MultiMetricMultiResolutionRegistration<TElastix>::BeforeRegistration()
{
  /** Get the components from this->m_Elastix and set them. */
  this->SetComponents();

  /** Set the number of resolutions.*/
  unsigned int numberOfResolutions = 3;
  this->m_Configuration->ReadParameter(numberOfResolutions, "NumberOfResolutions", 0);
  this->SetNumberOfLevels(numberOfResolutions);

  /** Set the FixedImageRegions to the buffered regions. */

  /** Make sure the fixed image is up to date. */
  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImages(); ++i)
  {
    try
    {
      this->GetElastix()->GetFixedImage(i)->Update();
    }
    catch (itk::ExceptionObject & excp)
    {
      /** Add information to the exception. */
      excp.SetLocation("MultiMetricMultiResolutionRegistration - BeforeRegistration()");
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while updating region info of the fixed image.\n";
      excp.SetDescription(err_str);
      /** Pass the exception to an higher level. */
      throw;
    }

    /** Set the fixedImageRegion. */
    this->SetFixedImageRegion(this->GetElastix()->GetFixedImage(i)->GetBufferedRegion(), i);
  }

  /** Add the target cells "Metric<i>" and "||Gradient<i>||" to IterationInfo
   * and format as floats.
   */
  const unsigned int nrOfMetrics = this->GetCombinationMetric()->GetNumberOfMetrics();
  unsigned int       width = 0;
  for (unsigned int i = nrOfMetrics; i > 0; i /= 10)
  {
    ++width;
  }
  for (unsigned int i = 0; i < nrOfMetrics; ++i)
  {
    std::ostringstream makestring1;
    makestring1 << "2:Metric" << std::setfill('0') << std::setw(width) << i;
    this->AddTargetCellToIterationInfo(makestring1.str().c_str());
    this->GetIterationInfoAt(makestring1.str().c_str()) << std::showpoint << std::fixed;

    std::ostringstream makestring2;
    makestring2 << "4:||Gradient" << std::setfill('0') << std::setw(width) << i << "||";
    this->AddTargetCellToIterationInfo(makestring2.str().c_str());
    this->GetIterationInfoAt(makestring2.str().c_str()) << std::showpoint << std::fixed;

    std::ostringstream makestring3;
    makestring3 << "Time" << std::setfill('0') << std::setw(width) << i << "[ms]";
    this->AddTargetCellToIterationInfo(makestring3.str().c_str());
    this->GetIterationInfoAt(makestring3.str().c_str()) << std::showpoint << std::fixed << std::setprecision(1);
  }

  /** Temporary? Use the multi-threaded version or not. */
  std::string tmp = this->m_Configuration->GetCommandLineArgument("-mtcombo");
  if (tmp == "true" || tmp.empty())
  {
    this->GetCombinationMetric()->SetUseMultiThread(true);
  }
  else
  {
    this->GetCombinationMetric()->SetUseMultiThread(false);
  }

} // end BeforeRegistration()


/**
 * ******************* AfterEachIteration ***********************
 */

template <class TElastix>
void
MultiMetricMultiResolutionRegistration<TElastix>::AfterEachIteration()
{
  /** Print the submetric values and gradients to IterationInfo. */
  const unsigned int nrOfMetrics = this->GetCombinationMetric()->GetNumberOfMetrics();
  unsigned int       width = 0;
  for (unsigned int i = nrOfMetrics; i > 0; i /= 10)
  {
    ++width;
  }
  for (unsigned int i = 0; i < nrOfMetrics; ++i)
  {
    std::ostringstream makestring1;
    makestring1 << "2:Metric" << std::setfill('0') << std::setw(width) << i;
    this->GetIterationInfoAt(makestring1.str().c_str()) << this->GetCombinationMetric()->GetMetricValue(i);

    std::ostringstream makestring2;
    makestring2 << "4:||Gradient" << std::setfill('0') << std::setw(width) << i << "||";
    this->GetIterationInfoAt(makestring2.str().c_str())
      << this->GetCombinationMetric()->GetMetricDerivativeMagnitude(i);

    std::ostringstream makestring3;
    makestring3 << "Time" << std::setfill('0') << std::setw(width) << i << "[ms]";
    this->GetIterationInfoAt(makestring3.str().c_str()) << this->GetCombinationMetric()->GetMetricComputationTime(i);
  }

  if (this->m_ShowExactMetricValue)
  {
    double currentExactMetricValue = 0.0;

    for (unsigned int i = 0; i < nrOfMetrics; ++i)
    {
      if (this->GetCombinationMetric()->GetUseMetric(i))
      {
        const double currentExactMetricValue_i = this->GetElastix()->GetElxMetricBase(i)->GetCurrentExactMetricValue();

        const double weight_i = this->GetCombinationMetric()->GetMetricWeight(i);

        currentExactMetricValue += weight_i * currentExactMetricValue_i;
      }
    }

    this->GetIterationInfoAt("ExactMetric") << currentExactMetricValue;
  }

} // end AfterEachIteration()


/**
 * ******************* BeforeEachResolution ***********************
 */

template <class TElastix>
void
MultiMetricMultiResolutionRegistration<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = this->GetCurrentLevel();

  /** Get the number of metrics. */
  unsigned int nrOfMetrics = this->GetCombinationMetric()->GetNumberOfMetrics();

  /** Set the masks in the metric. */
  this->UpdateFixedMasks(level);
  this->UpdateMovingMasks(level);

  /** Set the use of relative metric weights. */
  bool useRelativeWeights = false;
  this->GetConfiguration()->ReadParameter(useRelativeWeights, "UseRelativeWeights", 0);
  this->GetCombinationMetric()->SetUseRelativeWeights(useRelativeWeights);

  /** Set the metric weights. The default metric weight is 1.0 / nrOfMetrics. */
  if (!useRelativeWeights)
  {
    double defaultWeight = 1.0 / static_cast<double>(nrOfMetrics);
    for (unsigned int metricnr = 0; metricnr < nrOfMetrics; ++metricnr)
    {
      double             weight = defaultWeight;
      std::ostringstream makestring;
      makestring << "Metric" << metricnr << "Weight";
      this->GetConfiguration()->ReadParameter(weight, makestring.str(), "", level, 0);
      this->GetCombinationMetric()->SetMetricWeight(weight, metricnr);
    }
  }
  else
  {
    /** Set the relative metric weights.
     * The default relative metric weight is 1.0 / nrOfMetrics.
     */
    double defaultRelativeWeight = 1.0 / static_cast<double>(nrOfMetrics);
    for (unsigned int metricnr = 0; metricnr < nrOfMetrics; ++metricnr)
    {
      double             weight = defaultRelativeWeight;
      std::ostringstream makestring;
      makestring << "Metric" << metricnr << "RelativeWeight";
      this->GetConfiguration()->ReadParameter(weight, makestring.str(), "", level, 0);
      this->GetCombinationMetric()->SetMetricRelativeWeight(weight, metricnr);
    }
  }

  /** Set whether to use a specific metric. */
  for (unsigned int metricnr = 0; metricnr < nrOfMetrics; ++metricnr)
  {
    bool               use = true;
    std::ostringstream makestring;
    makestring << "Metric" << metricnr << "Use";
    this->GetConfiguration()->ReadParameter(use, makestring.str(), "", level, 0, false);
    this->GetCombinationMetric()->SetUseMetric(use, metricnr);
  }

  /** Check if the exact metric value, computed on all pixels, should be shown.
   * If at least one of the metrics has it enabled, show also the weighted sum of all
   * exact metric values. */

  /** Show the exact metric in every iteration? */
  this->m_ShowExactMetricValue = false;
  for (unsigned int metricnr = 0; metricnr < nrOfMetrics; ++metricnr)
  {
    this->m_ShowExactMetricValue |= this->GetElastix()->GetElxMetricBase(metricnr)->GetShowExactMetricValue();
  }

  if (this->m_ShowExactMetricValue)
  {
    /** Define the name of the ExactMetric column */
    std::string exactMetricColumn = "ExactMetric";

    /** Remove the ExactMetric-column, if it already existed. */
    this->RemoveTargetCellFromIterationInfo(exactMetricColumn.c_str());

    /** Create a new column in the iteration info table */
    this->AddTargetCellToIterationInfo(exactMetricColumn.c_str());
    this->GetIterationInfoAt(exactMetricColumn.c_str()) << std::showpoint << std::fixed;
  }

} // end BeforeEachResolution()


/**
 * *********************** SetComponents ************************
 */

template <class TElastix>
void
MultiMetricMultiResolutionRegistration<TElastix>::SetComponents()
{
  /** Get the component from this->GetElastix() (as elx::...BaseType *),
   * cast it to the appropriate type and set it in 'this'.
   */
  const unsigned int nrOfMetrics = this->GetElastix()->GetNumberOfMetrics();
  this->GetCombinationMetric()->SetNumberOfMetrics(nrOfMetrics);
  for (unsigned int i = 0; i < nrOfMetrics; ++i)
  {
    this->GetCombinationMetric()->SetMetric(this->GetElastix()->GetElxMetricBase(i)->GetAsITKBaseType(), i);
  }

  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImages(); ++i)
  {
    this->SetFixedImage(this->GetElastix()->GetFixedImage(i), i);
  }

  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfMovingImages(); ++i)
  {
    this->SetMovingImage(this->GetElastix()->GetMovingImage(i), i);
  }

  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImagePyramids(); ++i)
  {
    this->SetFixedImagePyramid(this->GetElastix()->GetElxFixedImagePyramidBase(i)->GetAsITKBaseType(), i);
  }

  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfMovingImagePyramids(); ++i)
  {
    this->SetMovingImagePyramid(this->GetElastix()->GetElxMovingImagePyramidBase(i)->GetAsITKBaseType(), i);
  }

  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfInterpolators(); ++i)
  {
    this->SetInterpolator(this->GetElastix()->GetElxInterpolatorBase(i)->GetAsITKBaseType(), i);
  }

  this->SetOptimizer(dynamic_cast<OptimizerType *>(this->GetElastix()->GetElxOptimizerBase()->GetAsITKBaseType()));

  this->SetTransform(this->GetElastix()->GetElxTransformBase()->GetAsITKBaseType());

  /** Samplers are not always needed: */
  for (unsigned int i = 0; i < nrOfMetrics; ++i)
  {
    if (this->GetElastix()->GetElxMetricBase(i)->GetAdvancedMetricUseImageSampler())
    {
      /** Try the i-th sampler for the i-th metric. */
      if (this->GetElastix()->GetElxImageSamplerBase(i))
      {
        this->GetElastix()->GetElxMetricBase(i)->SetAdvancedMetricImageSampler(
          this->GetElastix()->GetElxImageSamplerBase(i)->GetAsITKBaseType());
      }
      else
      {
        /** When a different fixed image pyramid is used for each metric,
         * using one sampler for all metrics makes no sense.
         */
        if (this->GetElastix()->GetElxFixedImagePyramidBase(i))
        {
          xl::xout["error"] << "ERROR: An ImageSamper for metric " << i << " must be provided!" << std::endl;
          itkExceptionMacro(<< "Not enough ImageSamplers provided!\nProvide an ImageSampler for metric " << i
                            << ", like:\n  (ImageSampler \"Random\" ... \"Random\")");
        }

        /** Try the zeroth image sampler for each metric. */
        if (this->GetElastix()->GetElxImageSamplerBase(0))
        {
          this->GetElastix()->GetElxMetricBase(i)->SetAdvancedMetricImageSampler(
            this->GetElastix()->GetElxImageSamplerBase(0)->GetAsITKBaseType());
        }
        else
        {
          xl::xout["error"] << "ERROR: No ImageSampler has been specified." << std::endl;
          itkExceptionMacro(<< "One of the metrics requires an ImageSampler, but it is not available!");
        }
      }

    } // if sampler required by metric
  }   // for loop over metrics

} // end SetComponents()


/**
 * ************************* UpdateFixedMasks ************************
 */

template <class TElastix>
void
MultiMetricMultiResolutionRegistration<TElastix>::UpdateFixedMasks(unsigned int level)
{
  /** some shortcuts */
  const unsigned int nrOfMetrics = this->GetElastix()->GetNumberOfMetrics();
  const unsigned int nrOfFixedMasks = this->GetElastix()->GetNumberOfFixedMasks();
  const unsigned int nrOfFixedImages = this->GetElastix()->GetNumberOfFixedImages();
  const unsigned int nrOfFixedImagePyramids = this->GetElastix()->GetNumberOfFixedImagePyramids();

  /** Array of bools, that remembers for each mask if erosion is wanted. */
  UseMaskErosionArrayType useMaskErosionArray;

  /** Bool that remembers if mask erosion is wanted in any of the masks
   * remains false when no masks are used.
   */
  bool useMaskErosion;

  /** Read whether mask erosion is wanted, if any masks were supplied. */
  useMaskErosion = this->ReadMaskParameters(useMaskErosionArray, nrOfFixedMasks, "Fixed", level);

  /** Create and start timer, to time the whole mask configuration procedure. */
  itk::TimeProbe timer;
  timer.Start();

  /** Now set the masks. */
  if (((nrOfFixedImages == 1) || (nrOfFixedMasks == 0)) && (nrOfFixedMasks <= 1) &&
      ((nrOfFixedImagePyramids == 1) || !useMaskErosion || (nrOfFixedMasks == 0)))
  {
    /** 1 image || nomask, <=1 mask, 1 pyramid || noerosion || nomask:
     * --> we can use one mask for all metrics! (or no mask at all).
     */
    FixedMaskSpatialObjectPointer fixedMask = this->GenerateFixedMaskSpatialObject(
      this->GetElastix()->GetFixedMask(), useMaskErosion, this->GetFixedImagePyramid(), level);
    this->GetCombinationMetric()->SetFixedImageMask(fixedMask);
  }
  else if ((nrOfFixedImages == 1) && (nrOfFixedMasks == 1))
  {
    /** 1 image, 1 mask, erosion && multiple pyramids
     * Set a differently eroded mask in each metric. The eroded
     * masks are all based on the same mask image, but generated with
     * different pyramid settings.
     */
    for (unsigned int i = 0; i < nrOfMetrics; ++i)
    {
      FixedMaskSpatialObjectPointer fixedMask = this->GenerateFixedMaskSpatialObject(
        this->GetElastix()->GetFixedMask(), useMaskErosion, this->GetFixedImagePyramid(i), level);
      this->GetCombinationMetric()->SetFixedImageMask(fixedMask, i);
    }
  }
  else
  {
    /** All other cases. Note that the number of pyramids should equal 1 or
     * should equal the number of metrics.
     * Set each supplied mask in its corresponding metric, possibly after erosion.
     * If more metrics than masks are present, the last metrics will not use a mask.
     * If less metrics than masks are present, the last masks will be ignored.
     */
    for (unsigned int i = 0; i < nrOfMetrics; ++i)
    {
      bool useMask_i = false; // default value in case of more metrics than masks
      if (i < nrOfFixedMasks)
      {
        useMask_i = useMaskErosionArray[i];
      }
      FixedImagePyramidPointer pyramid_i = this->GetFixedImagePyramid(); // default value in case of only 1 pyramid
      if (i < nrOfFixedImagePyramids)
      {
        pyramid_i = this->GetFixedImagePyramid(i);
      }
      FixedMaskSpatialObjectPointer fixedMask =
        this->GenerateFixedMaskSpatialObject(this->GetElastix()->GetFixedMask(i), useMask_i, pyramid_i, level);
      this->GetCombinationMetric()->SetFixedImageMask(fixedMask, i);
    }
  } // end else

  /** Stop timer and print the elapsed time. */
  timer.Stop();
  elxout << "Setting the fixed masks took: " << static_cast<long>(timer.GetMean() * 1000) << " ms." << std::endl;

} // end UpdateFixedMasks()


/**
 * ************************* UpdateMovingMasks ************************
 */

template <class TElastix>
void
MultiMetricMultiResolutionRegistration<TElastix>::UpdateMovingMasks(unsigned int level)
{
  /** Some shortcuts. */
  const unsigned int nrOfMetrics = this->GetElastix()->GetNumberOfMetrics();
  const unsigned int nrOfMovingMasks = this->GetElastix()->GetNumberOfMovingMasks();
  const unsigned int nrOfMovingImages = this->GetElastix()->GetNumberOfMovingImages();
  const unsigned int nrOfMovingImagePyramids = this->GetElastix()->GetNumberOfMovingImagePyramids();

  /** Array of bools, that remembers for each mask if erosion is wanted. */
  UseMaskErosionArrayType useMaskErosionArray;

  /** Bool that remembers if mask erosion is wanted in any of the masks
   * remains false when no masks are used.
   */
  bool useMaskErosion;

  /** Read whether mask erosion is wanted, if any masks were supplied. */
  useMaskErosion = this->ReadMaskParameters(useMaskErosionArray, nrOfMovingMasks, "Moving", level);

  /** Create and start timer, to time the whole mask configuration procedure. */
  itk::TimeProbe timer;
  timer.Start();

  /** Now set the masks. */
  if (((nrOfMovingImages == 1) || (nrOfMovingMasks == 0)) && (nrOfMovingMasks <= 1) &&
      ((nrOfMovingImagePyramids == 1) || !useMaskErosion || (nrOfMovingMasks == 0)))
  {
    /** 1 image || nomask, <=1 mask, 1 pyramid || noerosion || nomask:
     * --> we can use one mask for all metrics! (or no mask at all).
     */
    MovingMaskSpatialObjectPointer movingMask = this->GenerateMovingMaskSpatialObject(
      this->GetElastix()->GetMovingMask(), useMaskErosion, this->GetMovingImagePyramid(), level);
    this->GetCombinationMetric()->SetMovingImageMask(movingMask);
  }
  else if ((nrOfMovingImages == 1) && (nrOfMovingMasks == 1))
  {
    /** 1 image, 1 mask, erosion && multiple pyramids
     * Set a differently eroded mask in each metric. The eroded
     * masks are all based on the same mask image, but generated with
     * different pyramid settings.
     */
    for (unsigned int i = 0; i < nrOfMetrics; ++i)
    {
      MovingMaskSpatialObjectPointer movingMask = this->GenerateMovingMaskSpatialObject(
        this->GetElastix()->GetMovingMask(), useMaskErosion, this->GetMovingImagePyramid(i), level);
      this->GetCombinationMetric()->SetMovingImageMask(movingMask, i);
    }
  }
  else
  {
    /** All other cases. Note that the number of pyramids should equal 1 or
     * should equal the number of metrics.
     * Set each supplied mask in its corresponding metric, possibly after erosion.
     * If more metrics than masks are present, the last metrics will not use a mask.
     * If less metrics than masks are present, the last masks will be ignored.
     */
    for (unsigned int i = 0; i < nrOfMetrics; ++i)
    {
      bool useMask_i = false; // default value in case of more metrics than masks
      if (i < nrOfMovingMasks)
      {
        useMask_i = useMaskErosionArray[i];
      }
      MovingImagePyramidPointer pyramid_i = this->GetMovingImagePyramid(); // default value in case of only 1 pyramid
      if (i < nrOfMovingImagePyramids)
      {
        pyramid_i = this->GetMovingImagePyramid(i);
      }
      MovingMaskSpatialObjectPointer movingMask =
        this->GenerateMovingMaskSpatialObject(this->GetElastix()->GetMovingMask(i), useMask_i, pyramid_i, level);
      this->GetCombinationMetric()->SetMovingImageMask(movingMask, i);
    }
  } // end else

  /** Stop timer and print the elapsed time. */
  timer.Stop();
  elxout << "Setting the moving masks took: " << static_cast<long>(timer.GetMean() * 1000) << " ms." << std::endl;

} // end UpdateMovingMasks()


} // end namespace elastix

#endif // end #ifndef elxMultiMetricMultiResolutionRegistration_hxx
