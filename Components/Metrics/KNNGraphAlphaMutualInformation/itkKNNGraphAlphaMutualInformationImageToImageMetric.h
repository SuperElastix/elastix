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
#ifndef itkKNNGraphAlphaMutualInformationImageToImageMetric_h
#define itkKNNGraphAlphaMutualInformationImageToImageMetric_h

/** Includes for the Superclass. */
#include "itkMultiInputImageToImageMetricBase.h"

/** Includes for the kNN trees. */
#include "itkArray.h"
#include "itkListSampleCArray.h"
#include "itkBinaryTreeBase.h"
#include "itkBinaryTreeSearchBase.h"

/** Supported trees. */
#include "itkANNkDTree.h"
#include "itkANNbdTree.h"
#include "itkANNBruteForceTree.h"

/** Supported tree searchers. */
#include "itkANNStandardTreeSearch.h"
#include "itkANNFixedRadiusTreeSearch.h"
#include "itkANNPriorityTreeSearch.h"

/** Include for the spatial derivatives. */
#include "itkArray2D.h"

namespace itk
{
/**
 * \class KNNGraphAlphaMutualInformationImageToImageMetric
 *
 * \brief Computes similarity between two images to be registered.
 *
 * This metric computes the alpha-Mutual Information (aMI) between
 * two multi-channeled data sets. Said otherwise, given two sets of
 * features, the aMI between them is calculated.
 * Since for higher dimensional aMI it is infeasible to compute high
 * dimensional joint histograms, here we adopt a framework based on
 * the length of certain graphs, see Neemuchwala. Specifically, we use
 * the k-Nearest Neighbour (kNN) graph, using an implementation provided
 * by the Approximate Nearest Neighbour (ANN) software package.
 *
 * Note that the feature image are given beforehand, and that values
 * are calculated by interpolation on the transformed point. For some
 * features, it would be better (but slower) to first apply the transform
 * on the image and then recalculate the feature.
 *
 * All the technical details can be found in:\n
 * M. Staring, U.A. van der Heide, S. Klein, M.A. Viergever and J.P.W. Pluim,
 * "Registration of Cervical MRI Using Multifeature Mutual Information,"
 * IEEE Transactions on Medical Imaging, vol. 28, no. 9, pp. 1412 - 1421,
 * September 2009.
 *
 * \ingroup RegistrationMetrics
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT KNNGraphAlphaMutualInformationImageToImageMetric
  : public MultiInputImageToImageMetricBase<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(KNNGraphAlphaMutualInformationImageToImageMetric);

  /** Standard itk. */
  using Self = KNNGraphAlphaMutualInformationImageToImageMetric;
  using Superclass = MultiInputImageToImageMetricBase<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(KNNGraphAlphaMutualInformationImageToImageMetric, MultiInputImageToImageMetricBase);

  /** Typedefs from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::RealType;
  using typename Superclass::GradientPixelType;
  using typename Superclass::GradientImageType;
  using typename Superclass::GradientImagePointer;
  using typename Superclass::GradientImageFilterType;
  using typename Superclass::GradientImageFilterPointer;
  using typename Superclass::FixedImageMaskType;
  using typename Superclass::FixedImageMaskPointer;
  using typename Superclass::MovingImageMaskType;
  using typename Superclass::MovingImageMaskPointer;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::ParametersType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::MovingImageRegionType;
  using typename Superclass::ImageSamplerType;
  using typename Superclass::ImageSamplerPointer;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::FixedImageLimiterType;
  using typename Superclass::MovingImageLimiterType;
  using typename Superclass::FixedImageLimiterOutputType;
  using typename Superclass::MovingImageLimiterOutputType;
  using typename Superclass::NonZeroJacobianIndicesType;

  /** Typedef's for storing multiple inputs. */
  using typename Superclass::FixedImageVectorType;
  using typename Superclass::FixedImageMaskVectorType;
  using typename Superclass::FixedImageRegionVectorType;
  using typename Superclass::MovingImageVectorType;
  using typename Superclass::MovingImageMaskVectorType;
  using typename Superclass::InterpolatorVectorType;
  using typename Superclass::FixedImageInterpolatorVectorType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedefs for the samples. */
  using MeasurementVectorType = Array<double>;
  using MeasurementVectorValueType = typename MeasurementVectorType::ValueType;
  using ListSampleType = typename Statistics::ListSampleCArray<MeasurementVectorType, double>;
  using ListSamplePointer = typename ListSampleType::Pointer;

  /** Typedefs for trees. */
  using BinaryKNNTreeType = BinaryTreeBase<ListSampleType>;
  using BinaryKNNTreePointer = typename BinaryKNNTreeType::Pointer;
  using ANNkDTreeType = ANNkDTree<ListSampleType>;
  using ANNbdTreeType = ANNbdTree<ListSampleType>;
  using ANNBruteForceTreeType = ANNBruteForceTree<ListSampleType>;

  /** Typedefs for tree searchers. */
  using BinaryKNNTreeSearchType = BinaryTreeSearchBase<ListSampleType>;
  using BinaryKNNTreeSearchPointer = typename BinaryKNNTreeSearchType::Pointer;
  using ANNStandardTreeSearchType = ANNStandardTreeSearch<ListSampleType>;
  using ANNFixedRadiusTreeSearchType = ANNFixedRadiusTreeSearch<ListSampleType>;
  using ANNPriorityTreeSearchType = ANNPriorityTreeSearch<ListSampleType>;

  using IndexArrayType = typename BinaryKNNTreeSearchType::IndexArrayType;
  using DistanceArrayType = typename BinaryKNNTreeSearchType::DistanceArrayType;

  using DerivativeValueType = typename DerivativeType::ValueType;
  using TransformJacobianValueType = typename TransformJacobianType::ValueType;

  /**
   * *** Set trees: ***
   * Currently kd, bd, and brute force trees are supported.
   */

  /** Set ANNkDTree. */
  void
  SetANNkDTree(unsigned int bucketSize, std::string splittingRule);

  /** Set ANNkDTree. */
  void
  SetANNkDTree(unsigned int bucketSize,
               std::string  splittingRuleFixed,
               std::string  splittingRuleMoving,
               std::string  splittingRuleJoint);

  /** Set ANNbdTree. */
  void
  SetANNbdTree(unsigned int bucketSize, std::string splittingRule, std::string shrinkingRule);

  /** Set ANNbdTree. */
  void
  SetANNbdTree(unsigned int bucketSize,
               std::string  splittingRuleFixed,
               std::string  splittingRuleMoving,
               std::string  splittingRuleJoint,
               std::string  shrinkingRuleFixed,
               std::string  shrinkingRuleMoving,
               std::string  shrinkingRuleJoint);

  /** Set ANNBruteForceTree. */
  void
  SetANNBruteForceTree();

  /**
   * *** Set tree searchers: ***
   * Currently standard, fixed radius, and priority tree searchers are supported.
   */

  /** Set ANNStandardTreeSearch. */
  void
  SetANNStandardTreeSearch(unsigned int kNearestNeighbors, double errorBound);

  /** Set ANNFixedRadiusTreeSearch. */
  void
  SetANNFixedRadiusTreeSearch(unsigned int kNearestNeighbors, double errorBound, double squaredRadius);

  /** Set ANNPriorityTreeSearch. */
  void
  SetANNPriorityTreeSearch(unsigned int kNearestNeighbors, double errorBound);

  /**
   * *** Standard metric stuff: ***
   */

  /** Initialize the metric. */
  void
  Initialize() override;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & Derivative) const override;

  /** Get the value for single valued optimizers. */
  MeasureType
  GetValue(const TransformParametersType & parameters) const override;

  /** Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                Derivative) const override;

  /** Set alpha from alpha - mutual information. */
  itkSetClampMacro(Alpha, double, 0.0, 1.0);

  /** Get alpha from alpha - mutual information. */
  itkGetConstReferenceMacro(Alpha, double);

  /** Avoid division by a small number. */
  itkSetClampMacro(AvoidDivisionBy, double, 0.0, 1.0);

  /** Avoid division by a small number. */
  itkGetConstReferenceMacro(AvoidDivisionBy, double);

protected:
  /** Constructor. */
  KNNGraphAlphaMutualInformationImageToImageMetric();

  /** Destructor. */
  ~KNNGraphAlphaMutualInformationImageToImageMetric() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Member variables. */
  BinaryKNNTreePointer m_BinaryKNNTreeFixed;
  BinaryKNNTreePointer m_BinaryKNNTreeMoving;
  BinaryKNNTreePointer m_BinaryKNNTreeJoint;

  BinaryKNNTreeSearchPointer m_BinaryKNNTreeSearcherFixed;
  BinaryKNNTreeSearchPointer m_BinaryKNNTreeSearcherMoving;
  BinaryKNNTreeSearchPointer m_BinaryKNNTreeSearcherJoint;

  double m_Alpha;
  double m_AvoidDivisionBy;

private:
  /** Typedef's for the computation of the derivative. */
  using typename Superclass::FixedImagePointType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageDerivativeType;
  using typename Superclass::MovingImageContinuousIndexType;
  using TransformJacobianContainerType = std::vector<TransformJacobianType>;
  // typedef std::vector<ParameterIndexArrayType>           TransformJacobianIndicesContainerType;
  using TransformJacobianIndicesContainerType = std::vector<NonZeroJacobianIndicesType>;
  using SpatialDerivativeType = Array2D<double>;
  using SpatialDerivativeContainerType = std::vector<SpatialDerivativeType>;

  /** This function takes the fixed image samples from the ImageSampler
   * and puts them in the listSampleFixed, together with the fixed feature
   * image samples. Also the corresponding moving image values and moving
   * feature values are computed and put into listSampleMoving. The
   * concatenation is put into listSampleJoint.
   * If desired, i.e. if doDerivative is true, then also things needed to
   * compute the derivative of the cost function to the transform parameters
   * are computed:
   * - The sparse Jacobian of the transformation (dT/dmu).
   * - The spatial derivatives of the moving (feature) images (dm/dx).
   */
  void
  ComputeListSampleValuesAndDerivativePlusJacobian(const ListSamplePointer &               listSampleFixed,
                                                   const ListSamplePointer &               listSampleMoving,
                                                   const ListSamplePointer &               listSampleJoint,
                                                   const bool                              doDerivative,
                                                   TransformJacobianContainerType &        jacobians,
                                                   TransformJacobianIndicesContainerType & jacobiansIndices,
                                                   SpatialDerivativeContainerType &        spatialDerivatives) const;

  /** This function calculates the spatial derivative of the
   * featureNr feature image at the point mappedPoint.
   * \todo move this to base class.
   */
  virtual void
  EvaluateMovingFeatureImageDerivatives(const MovingImagePointType & mappedPoint,
                                        SpatialDerivativeType &      featureGradients) const;

  /** This function essentially computes D1 - D2, but also takes
   * care of going from a sparse matrix (hence the indices) to a
   * full sized matrix.
   */
  virtual void
  UpdateDerivativeOfGammas(const SpatialDerivativeType & D1sparse,
                           const SpatialDerivativeType & D2sparse_M,
                           const SpatialDerivativeType & D2sparse_J,
                           // const ParameterIndexArrayType & D1indices,
                           // const ParameterIndexArrayType & D2indices_M,
                           // const ParameterIndexArrayType & D2indices_J,
                           const NonZeroJacobianIndicesType & D1indices,
                           const NonZeroJacobianIndicesType & D2indices_M,
                           const NonZeroJacobianIndicesType & D2indices_J,
                           const MeasurementVectorType &      diff_M,
                           const MeasurementVectorType &      diff_J,
                           const MeasureType &                distance_M,
                           const MeasureType &                distance_J,
                           DerivativeType &                   dGamma_M,
                           DerivativeType &                   dGamma_J) const;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkKNNGraphAlphaMutualInformationImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkKNNGraphAlphaMutualInformationImageToImageMetric_h
