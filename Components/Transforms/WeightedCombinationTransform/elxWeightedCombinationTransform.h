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
#ifndef elxWeightedCombinationTransform_h
#define elxWeightedCombinationTransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkWeightedCombinationTransform.h"
#include "itkAdvancedCombinationTransform.h"

namespace elastix
{

/**
 * \class WeightedCombinationTransformElastix
 * \brief A transform based on the itk::WeightedCombinationTransform.
 *
 * This transform is a weighted combination transformation. It implements
 * \f$T(x) = \sum_i w_i T_i(x)\f$.
 *
 * The transformparameters are the weighting factors \f$w_i\f$ for each
 * subtransform \f$T_i(x)\f$. You could use this to implement registration
 * using a statistical deformation model. Each subtransform would then be
 * a principal component that follows from your statistical model for example.
 *
 * A normalization factor can optionally be used:
 * \f$T(x) = \sum_i w_i T_i(x) / \sum_i w_i\f$
 *
 * You can also use this class to average transformations found by previous
 * elastix runs.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "WeightedCombinationTransform")</tt>
 * \parameter NormalizeCombinationWeights: use the normalized expression
 * \f$T(x) = \sum_i w_i T_i(x) / \sum_i w_i \f$.\n
 *    <tt>(NormalizeCombinationWeights "true" )</tt>\n
 * Default value: "false". Different values in each resolution are not supported.
 * \parameter SubTransforms: a list of transform parameter filenames that
 * will serve as subtransforms \f$T_i(x)\f$.\n
 *    <tt>(SubTransforms "tp0.txt" "TransformParameters.1.txt" "tpbspline.txt" )</tt>\n
 * \parameter AutomaticScalesEstimation: if this parameter is set to "true" the Scales
 *    parameter is ignored and the scales are determined automatically. \n
 *    example: <tt>(AutomaticScalesEstimation "true") </tt> \n
 *    Default: "false".
 * \parameter Scales: The scale factor for each transform parameter, during optimization.\n
 *    If your input subtransforms have very different magnitudes, you may compensate for that
 *    by supplying scales, which will make the optimization CostFunction better behaving.
 *    For subtransforms with a high magnitude, provide a large scale then. NB: not in all cases
 *    you may want this.
 *    example: <tt>(Scales 1.0 1.0 10.0) </tt> \n
 *    Default: 1 for each parameter. See also AutomaticScalesEstimation, which is more convenient.
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter NormalizeCombinationWeights: use the normalized expression
 * \f$T(x) = \sum_i w_i T_i(x) / \sum_i w_i \f$.\n
 *    <tt>(NormalizeCombinationWeights "true" )</tt>\n
 * Default value: "false". Different values in each resolution are not supported.
 * \transformparameter SubTransforms: a list of transform parameter filenames that
 * will serve as subtransforms \f$T_i(x)\f$.\n
 *    <tt>(SubTransforms "tp0.txt" "TransformParameters.1.txt" "tpbspline.txt" )</tt>\n
 *
 * \ingroup Transforms
 * \sa WeightedCombinationTransform
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT WeightedCombinationTransformElastix
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef WeightedCombinationTransformElastix Self;

  typedef itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension>
    Superclass1;

  typedef elx::TransformBase<TElastix> Superclass2;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  typedef itk::WeightedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension,
                                            elx::TransformBase<TElastix>::MovingImageDimension>
    WeightedCombinationTransformType;

  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(WeightedCombinationTransformElastix, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "WeightedCombinationTransform")</tt>\n
   */
  elxClassNameMacro("WeightedCombinationTransform");

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::ScalarType                ScalarType;
  typedef typename Superclass1::ParametersType            ParametersType;
  typedef typename Superclass1::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass1::JacobianType              JacobianType;
  typedef typename Superclass1::InputVectorType           InputVectorType;
  typedef typename Superclass1::OutputVectorType          OutputVectorType;
  typedef typename Superclass1::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass1::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass1::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass1::OutputVnlVectorType       OutputVnlVectorType;
  typedef typename Superclass1::InputPointType            InputPointType;
  typedef typename Superclass1::OutputPointType           OutputPointType;

  /** Typedef's from the TransformBase class. */
  typedef typename Superclass2::ElastixType                ElastixType;
  typedef typename Superclass2::ElastixPointer             ElastixPointer;
  typedef typename Superclass2::ParameterMapType           ParameterMapType;
  typedef typename Superclass2::ConfigurationType          ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer       ConfigurationPointer;
  typedef typename Superclass2::RegistrationType           RegistrationType;
  typedef typename Superclass2::RegistrationPointer        RegistrationPointer;
  typedef typename Superclass2::CoordRepType               CoordRepType;
  typedef typename Superclass2::FixedImageType             FixedImageType;
  typedef typename Superclass2::MovingImageType            MovingImageType;
  typedef typename Superclass2::ITKBaseType                ITKBaseType;
  typedef typename Superclass2::CombinationTransformType   CombinationTransformType;
  typedef typename Superclass2::CommandLineArgumentMapType CommandLineArgumentMapType;
  typedef typename Superclass2::CommandLineEntryType       CommandLineEntryType;

  /** Extra typedefs */
  typedef typename WeightedCombinationTransformType::Pointer                WeightedCombinationTransformPointer;
  typedef typename WeightedCombinationTransformType::TransformContainerType TransformContainerType;
  typedef typename WeightedCombinationTransformType::TransformType          SubTransformType;
  typedef typename WeightedCombinationTransformType::TransformPointer       SubTransformPointer;

  /** For scales setting in the optimizer */
  typedef typename Superclass2::ScalesType ScalesType;

  /** Execute stuff before the actual registration:
   * \li Read some parameters
   * \li Call InitializeTransform.
   * \li Set the scales. */
  void
  BeforeRegistration(void) override;

  /** Initialize Transform.
   * \li Load subtransforms
   * \li Set all parameters to 1/NrOfSubTransforms (if NormalizeCombinationWeights=true)
   * or 0, (if NormalizeCombinationWeights=false)
   * \li Set the initial parameters in the Registration object
   * This function is called by BeforeRegistration().
   */
  virtual void
  InitializeTransform(void);

  /** Set the scales
   * \li If AutomaticScalesEstimation is "true" estimate scales
   * \li If scales are provided by the user use those,
   * \li Otherwise use some default value: 1.
   * This function is called by BeforeRegistration, after
   * the InitializeTransform function is called
   */
  virtual void
  SetScales(void);

  /** Function to read transform-parameters from a file.
   *
   * It loads the subtransforms, the NormalizeWeights option,
   * and calls the superclass' implementation.
   */
  void
  ReadFromFile(void) override;

  /** Load from the parameter file a list of subtransforms. The filenames are
   * stored in the m_SubTransformFileNames list */
  virtual void
  LoadSubTransforms(void);

protected:
  /** The constructor. */
  WeightedCombinationTransformElastix();
  /** The destructor. */
  ~WeightedCombinationTransformElastix() override = default;

  WeightedCombinationTransformPointer m_WeightedCombinationTransform;
  std::vector<std::string>            m_SubTransformFileNames;

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap(void) const override;

  /** The deleted copy constructor. */
  WeightedCombinationTransformElastix(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxWeightedCombinationTransform.hxx"
#endif

#endif // end #ifndef elxWeightedCombinationTransform_h
