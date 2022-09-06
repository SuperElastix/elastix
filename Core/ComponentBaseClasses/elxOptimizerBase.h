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

#ifndef elxOptimizerBase_h
#define elxOptimizerBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkOptimizer.h"

namespace elastix
{

/**
 * \class OptimizerBase
 * \brief This class is the elastix base class for all Optimizers.
 *
 * This class contains all the common functionality for Optimizers.
 *
 * The parameters used in this class are:
 * \parameter NewSamplesEveryIteration: if this flag is set to "true" some
 *    optimizers ask the metric to select a new set of spatial samples in
 *    every iteration. This, if used in combination with the correct optimizer (such as the
 *    StandardGradientDescent), and ImageSampler (Random, RandomCoordinate, or RandomSparseMask),
 *    allows for a very low number of spatial samples (around 2000), even with large images
 *    and transforms with a large number of parameters.\n
 *    Choose one from {"true", "false"} for every resolution.\n
 *    example: <tt>(NewSamplesEveryIteration "true" "true" "true")</tt> \n
 *    Default is "false" for every resolution.\n
 *
 * \ingroup Optimizers
 * \ingroup ComponentBaseClasses
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT OptimizerBase : public BaseComponentSE<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(OptimizerBase);

  /** Standard ITK-stuff. */
  using Self = OptimizerBase;
  using Superclass = BaseComponentSE<TElastix>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(OptimizerBase, BaseComponentSE);

  /** Typedefs inherited from Elastix. */
  using typename Superclass::ElastixType;
  using typename Superclass::RegistrationType;

  /** ITKBaseType. */
  using ITKBaseType = itk::Optimizer;

  /** Typedef needed for the SetCurrentPositionPublic function. */
  using ParametersType = typename ITKBaseType::ParametersType;

  /** Retrieves this object as ITKBaseType. */
  ITKBaseType *
  GetAsITKBaseType()
  {
    return &(this->GetSelf());
  }


  /** Retrieves this object as ITKBaseType, to use in const functions. */
  const ITKBaseType *
  GetAsITKBaseType() const
  {
    return &(this->GetSelf());
  }


  /** Add empty SetCurrentPositionPublic, so this function is known in every inherited class. */
  virtual void
  SetCurrentPositionPublic(const ParametersType & param);

  /** Execute stuff before each new pyramid resolution:
   * \li Find out if new samples are used every new iteration in this resolution.
   */
  void
  BeforeEachResolutionBase() override;

  /** Execute stuff after registration:
   * \li Compute and print MD5 hash of the transform parameters.
   */
  void
  AfterRegistrationBase() override;

  /** Method that sets the scales defined by a sinus
   * scale[i] = amplitude^( sin(i/nrofparam*2pi*frequency) )
   */
  virtual void
  SetSinusScales(double amplitude, double frequency, unsigned long numberOfParameters);

protected:
  /** The constructor. */
  OptimizerBase() = default;
  /** The destructor. */
  ~OptimizerBase() override = default;

  /** Force the metric to base its computation on a new subset of image samples.
   * Not every metric may have implemented this.
   */
  virtual void
  SelectNewSamples();

  /** Check whether the user asked to select new samples every iteration. */
  virtual bool
  GetNewSamplesEveryIteration() const;

private:
  elxDeclarePureVirtualGetSelfMacro(ITKBaseType);

  /** Member variable to store the user preference for using new
   * samples each iteration.
   */
  bool m_NewSamplesEveryIteration{ false };
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxOptimizerBase.hxx"
#endif

#endif // end #ifndef elxOptimizerBase_h
