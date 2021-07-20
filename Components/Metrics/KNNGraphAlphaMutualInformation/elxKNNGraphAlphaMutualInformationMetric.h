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
#ifndef elxKNNGraphAlphaMutualInformationMetric_h
#define elxKNNGraphAlphaMutualInformationMetric_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkKNNGraphAlphaMutualInformationImageToImageMetric.h"

namespace elastix
{

/**
 * \class KNNGraphAlphaMutualInformationMetric
 * \brief A metric based on the
 * itk::KNNGraphAlphaMutualInformationImageToImageMetric.
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "KNNGraphAlphaMutualInformation")</tt>
 * \parameter Alpha: since this metric calculates alpha - mutual information. \n
 *    <tt>(Alpha 0.5)</tt> \n
 *    Choose a value between 0.0 and 1.0. The default is 0.5.
 * \parameter TreeType: The type of the kNN binary tree. \n
 *    <tt>(TreeType "BDTree" "BruteForceTree")</tt> \n
 *    Choose one of { KDTree, BDTree, BruteForceTree }. \n
 *    The default is "KDTree" for all resolutions.
 * \parameter BucketSize: The maximum number of samples in one bucket. \n
 *    This parameter influences the calculation time only, and is not appropiate for the BruteForceTree. \n
 *    <tt>(BucketSize 5 100 50)</tt> \n
 *    The default is 50 for all resolutions.
 * \parameter SplittingRule: This rule defines how the feature space is split. \n
 *    <tt>(SplittingRule "ANN_KD_STD" "ANN_KD_FAIR")</tt> \n
 *    Choose one of { ANN_KD_STD, ANN_KD_MIDPT, ANN_KD_SL_MIDPT, ANN_KD_FAIR, ANN_KD_SL_FAIR, ANN_KD_SUGGEST } \n
 *    The default is "ANN_KD_SL_MIDPT" for all resolutions.
 * \parameter ShrinkingRule: This rule defines how the feature space is shrinked. \n
 *    <tt>(ShrinkingRule "ANN_BD_CENTROID" "ANN_BD_NONE")</tt> \n
 *    Choose one of { ANN_BD_NONE, ANN_BD_SIMPLE, ANN_BD_CENTROID, ANN_BD_SUGGEST } \n
 *    The default is "ANN_BD_SIMPLE" for all resolutions.
 * \parameter TreeSearchType: The type of the binary tree searcher. \n
 *    <tt>(TreeSearchType "Standard" "FixedRadius")</tt> \n
 *    Choose one of { Standard, FixedRadius, Priority } \n
 *    The default is "Standard" for all resolutions.
 * \parameter KNearestNeighbours: The number of nearest neighbours to be searched. \n
 *    <tt>(KNearestNeighbours 50 20 35)</tt> \n
 *    The default is 20 for all resolutions.
 * \parameter ErrorBound: error accepted in finding the nearest neighbours. \n
 *    An ErrorBound of 0.0 equals exact searching, higher error bounds should
 *    result in smaller computation times. \n
 *    <tt>(ErrorBound 32.0 8.0 0.0)</tt> \n
 *    The default is 0.0 for all resolutions.
 * \parameter SquaredSearchRadius: the radius of the sphere where there is searched for neighbours. \n
 *    This option is only appropiate for FixedRadius search. \n
 *    <tt>(SquaredSearchRadius 32.0 8.0 8.0)</tt> \n
 *    The default is 0.0 for all resolutions, which means no radius.
 * \parameter AvoidDivisionBy: a small number to avoid division by zero in the implentation. \n
 *    <tt>(AvoidDivisionBy 0.000000001)</tt> \n
 *    The default is 1e-5.
 *
 * \warning Note that we assume the FixedFeatureImageType to have the same
 * pixeltype as the FixedImageType
 *
 * \sa KNNGraphAlphaMutualInformationImageToImageMetric, ParzenWindowMutualInformationImageToImageMetric
 * \ingroup Metrics
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT KNNGraphAlphaMutualInformationMetric
  : public itk::KNNGraphAlphaMutualInformationImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                                 typename MetricBase<TElastix>::MovingImageType>
  , public MetricBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef KNNGraphAlphaMutualInformationMetric Self;
  typedef itk::KNNGraphAlphaMutualInformationImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                                typename MetricBase<TElastix>::MovingImageType>
                                        Superclass1;
  typedef MetricBase<TElastix>          Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(KNNGraphAlphaMutualInformationMetric, itk::KNNGraphAlphaMutualInformationImageToImageMetric);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "KNNGraphAlphaMutualInformation")</tt>\n
   */
  elxClassNameMacro("KNNGraphAlphaMutualInformation");

  /** Typedefs inherited from the superclass.*/
  typedef typename Superclass1::TransformType           TransformType;
  typedef typename Superclass1::TransformPointer        TransformPointer;
  typedef typename Superclass1::TransformJacobianType   TransformJacobianType;
  typedef typename Superclass1::InterpolatorType        InterpolatorType;
  typedef typename Superclass1::MeasureType             MeasureType;
  typedef typename Superclass1::DerivativeType          DerivativeType;
  typedef typename Superclass1::ParametersType          ParametersType;
  typedef typename Superclass1::FixedImageType          FixedImageType;
  typedef typename Superclass1::MovingImageType         MovingImageType;
  typedef typename Superclass1::FixedImageConstPointer  FixedImageConstPointer;
  typedef typename Superclass1::MovingImageConstPointer MovingImageConstPointer;

  /** The fixed image dimension */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedef's inherited from Elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Typedefs for feature images. */
  typedef FixedImageType  FixedFeatureImageType;
  typedef MovingImageType MovingFeatureImageType;

  /** Execute stuff before the registration:
   * \li Set the alpha from alpha - MI.
   * \li Set the number of fixed feature images.
   * \li Set the number of moving feature images.
   * \li Set the fixed feature images filenames.
   * \li Set the moving feature images filenames.
   * \li Set the spline orders of the fixed feature interpolators.
   * \li Set the spline orders of the moving feature interpolators.
   */
  void
  BeforeRegistration(void) override;

  /** Execute stuff before each new pyramid resolution:
   * \li Set the tree type.
   * \li Set the bucket size, if appropriate.
   * \li Set the splitting rule, if appropriate.
   * \li Set the shrinking rule, if appropriate.
   * \li Set the tree searcher type.
   * \li Set the k NearestNeighbours.
   * \li Set the error bound epsilon for ANN search.
   * \li Set the squared search radius, if appropriate.
   */
  void
  BeforeEachResolution(void) override;

  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  void
  Initialize(void) override;

protected:
  /** The constructor. */
  KNNGraphAlphaMutualInformationMetric() = default;
  /** The destructor. */
  ~KNNGraphAlphaMutualInformationMetric() override = default;

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  KNNGraphAlphaMutualInformationMetric(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxKNNGraphAlphaMutualInformationMetric.hxx"
#endif

#endif // end #ifndef elxKNNGraphAlphaMutualInformationMetric_h
