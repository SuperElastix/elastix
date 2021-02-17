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
#ifndef elxRayCastResampleInterpolator_h
#define elxRayCastResampleInterpolator_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedRayCastInterpolateImageFunction.h"
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedTransform.h"
#include "itkEulerTransform.h"

namespace elastix
{

/**
 * \class RayCastResampleInterpolator
 * \brief An interpolator based on ...
 *
 * \ingroup Interpolators
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT RayCastResampleInterpolator
  : public itk::AdvancedRayCastInterpolateImageFunction<typename ResampleInterpolatorBase<TElastix>::InputImageType,
                                                        typename ResampleInterpolatorBase<TElastix>::CoordRepType>
  , public ResampleInterpolatorBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef RayCastResampleInterpolator Self;
  typedef itk::AdvancedRayCastInterpolateImageFunction<typename ResampleInterpolatorBase<TElastix>::InputImageType,
                                                       typename ResampleInterpolatorBase<TElastix>::CoordRepType>
                                             Superclass1;
  typedef ResampleInterpolatorBase<TElastix> Superclass2;
  typedef itk::SmartPointer<Self>            Pointer;
  typedef itk::SmartPointer<const Self>      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayCastResampleInterpolator, AdvancedRayCastInterpolateImageFunction);

  /** Name of this class.
   * Use this name in the parameter file to select this specific resample interpolator. \n
   * example: <tt>(ResampleInterpolator "FinalRayCastInterpolator")</tt>\n
   */
  elxClassNameMacro("FinalRayCastInterpolator");

  /** Dimension of the image. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass1::ImageDimension);

  /** Typedef's inherited from the superclass. */
  typedef typename Superclass1::OutputType          OutputType;
  typedef typename Superclass1::InputImageType      InputImageType;
  typedef typename Superclass1::IndexType           IndexType;
  typedef typename Superclass1::ContinuousIndexType ContinuousIndexType;
  typedef typename Superclass1::PointType           PointType;
  typedef typename Superclass1::SizeType            SizeType;
  typedef typename InputImageType::SpacingType      SpacingType;

  /** Typedef's from ResampleInterpolatorBase. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Typedef's for CombinationTransform */
  typedef typename itk::EulerTransform<typename InterpolatorBase<TElastix>::CoordRepType, ImageDimension>
                                                      EulerTransformType;
  typedef typename EulerTransformType::ParametersType TransformParametersType;
  typedef typename EulerTransformType::Pointer        EulerTransformPointer;
  typedef typename itk::AdvancedTransform<typename InterpolatorBase<TElastix>::CoordRepType,
                                          itkGetStaticConstMacro(ImageDimension),
                                          itkGetStaticConstMacro(ImageDimension)>
                                                  AdvancedTransformType;
  typedef typename AdvancedTransformType::Pointer AdvancedTransformPointer;
  typedef typename itk::AdvancedCombinationTransform<typename InterpolatorBase<TElastix>::CoordRepType,
                                                     itkGetStaticConstMacro(ImageDimension)>
                                                     CombinationTransformType;
  typedef typename CombinationTransformType::Pointer CombinationTransformPointer;

  typedef typename Superclass2::ParameterMapType ParameterMapType;

  int
  BeforeAll(void) override;

  void
  BeforeRegistration(void) override;

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile(void) override;

protected:
  /** The constructor. */
  RayCastResampleInterpolator() = default;

  /** The destructor. */
  ~RayCastResampleInterpolator() override = default;

  /** Helper function to initialize the combination transform
   * with a pre-transform.
   */
  void
  InitializeRayCastInterpolator(void);

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) interpolator type. */
  ParameterMapType
  CreateDerivedTransformParametersMap() const override;

  /** The deleted copy constructor. */
  RayCastResampleInterpolator(const Self &) = delete;

  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;

  EulerTransformPointer       m_PreTransform;
  TransformParametersType     m_PreParameters;
  CombinationTransformPointer m_CombinationTransform;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxRayCastResampleInterpolator.hxx"
#endif

#endif // end elxRayCastResampleInterpolator_h
