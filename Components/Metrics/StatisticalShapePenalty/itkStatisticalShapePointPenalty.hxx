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
#ifndef itkStatisticalShapePointPenalty_hxx
#define itkStatisticalShapePointPenalty_hxx

#include "itkStatisticalShapePointPenalty.h"
#include <cmath>

namespace itk
{
/**
 * ******************* Constructor *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::StatisticalShapePointPenalty()
{
  this->m_MeanVector = nullptr;
  this->m_EigenVectors = nullptr;
  this->m_EigenValues = nullptr;
  this->m_EigenValuesRegularized = nullptr;
  this->m_ProposalDerivative = nullptr;
  this->m_InverseCovarianceMatrix = nullptr;

  this->m_ShrinkageIntensityNeedsUpdate = true;
  this->m_BaseVarianceNeedsUpdate = true;
  this->m_VariancesNeedsUpdate = true;

} // end Constructor


/**
 * ******************* Destructor *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::~StatisticalShapePointPenalty()
{
  if (this->m_MeanVector != nullptr)
  {
    delete this->m_MeanVector;
    this->m_MeanVector = nullptr;
  }
  if (this->m_CovarianceMatrix != nullptr)
  {
    delete this->m_CovarianceMatrix;
    this->m_CovarianceMatrix = nullptr;
  }
  if (this->m_EigenVectors != nullptr)
  {
    delete this->m_EigenVectors;
    this->m_EigenVectors = nullptr;
  }
  if (this->m_EigenValues != nullptr)
  {
    delete this->m_EigenValues;
    this->m_EigenValues = nullptr;
  }
  if (this->m_EigenValuesRegularized != nullptr)
  {
    delete this->m_EigenValuesRegularized;
    this->m_EigenValuesRegularized = nullptr;
  }
  if (this->m_ProposalDerivative != nullptr)
  {
    delete this->m_ProposalDerivative;
    this->m_ProposalDerivative = nullptr;
  }
  if (this->m_InverseCovarianceMatrix != nullptr)
  {
    delete this->m_InverseCovarianceMatrix;
    this->m_InverseCovarianceMatrix = nullptr;
  }

} // end Destructor


/**
 * *********************** Initialize *****************************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::Initialize()
{
  /** Call the initialize of the superclass. */
  this->Superclass::Initialize();

  const unsigned int shapeLength = Self::FixedPointSetDimension * this->GetFixedPointSet()->GetNumberOfPoints();
  if (this->m_NormalizedShapeModel)
  {
    this->m_ProposalLength = shapeLength + Self::FixedPointSetDimension + 1;

    /** Automatic selection of regularization variances. */
    if (this->m_BaseVariance == -1.0 || this->m_CentroidXVariance == -1.0 || this->m_CentroidYVariance == -1.0 ||
        this->m_CentroidZVariance == -1.0 || this->m_SizeVariance == -1.0)
    {
      vnl_vector<double> covDiagonal = this->m_CovarianceMatrix->get_diagonal();
      if (this->m_BaseVariance == -1.0)
      {
        this->m_BaseVariance = covDiagonal.extract(shapeLength).mean();
      }
      if (this->m_CentroidXVariance == -1.0)
      {
        this->m_CentroidXVariance = covDiagonal.get(shapeLength);
      }
      if (this->m_CentroidYVariance == -1.0)
      {
        this->m_CentroidYVariance = covDiagonal.get(shapeLength + 1);
      }
      if (this->m_CentroidZVariance == -1.0)
      {
        this->m_CentroidZVariance = covDiagonal.get(shapeLength + 2);
      }
      if (this->m_SizeVariance == -1.0)
      {
        this->m_SizeVariance = covDiagonal.get(shapeLength + 3);
      }
    } // End automatic selection of regularization variances.
  }
  else
  {
    this->m_ProposalLength = shapeLength;
    /** Automatic selection of regularization variances. */
    if (this->m_BaseVariance == -1.0)
    {
      vnl_vector<double> covDiagonal = this->m_CovarianceMatrix->get_diagonal();
      this->m_BaseVariance = covDiagonal.extract(shapeLength).mean();
    } // End automatic selection of regularization variances.
  }

  switch (this->m_ShapeModelCalculation)
  {
    case 0: // full covariance
    {
      if (this->m_ShrinkageIntensityNeedsUpdate || this->m_BaseVarianceNeedsUpdate ||
          (this->m_NormalizedShapeModel && this->m_VariancesNeedsUpdate))
      {
        vnl_matrix<double> regularizedCovariance = (1 - this->m_ShrinkageIntensity) * (*this->m_CovarianceMatrix);
        vnl_vector<double> regCovDiagonal = regularizedCovariance.get_diagonal();
        if (this->m_NormalizedShapeModel)
        {
          regCovDiagonal.update(this->m_ShrinkageIntensity * this->m_BaseVariance +
                                regCovDiagonal.extract(shapeLength));
          regCovDiagonal[shapeLength] += this->m_ShrinkageIntensity * this->m_CentroidXVariance;
          regCovDiagonal[shapeLength + 1] += this->m_ShrinkageIntensity * this->m_CentroidYVariance;
          regCovDiagonal[shapeLength + 2] += this->m_ShrinkageIntensity * this->m_CentroidZVariance;
          regCovDiagonal[shapeLength + 3] += this->m_ShrinkageIntensity * this->m_SizeVariance;
        }
        else
        {
          regCovDiagonal += this->m_ShrinkageIntensity * this->m_BaseVariance;
        }
        regularizedCovariance.set_diagonal(regCovDiagonal);
        /** If no regularization is applied, the user is responsible for providing an
         * invertible Covariance Matrix. For a Moore-Penrose pseudo inverse use
         * ShrinkageIntensity=0 and ShapeModelCalculation=1 or 2.
         */
        this->m_InverseCovarianceMatrix = new vnl_matrix<double>(vnl_svd_inverse(regularizedCovariance));
      }
      this->m_EigenValuesRegularized = nullptr;
      break;
    }
    case 1: // decomposed covariance (uniform regularization)
    {
      if (this->m_NormalizedShapeModel == true)
      {
        itkExceptionMacro(<< "ShapeModelCalculation option 1 is only implemented for NormalizedShapeModel = false");
      }

      PCACovarianceType                pcaCovariance(*this->m_CovarianceMatrix);
      typename VnlVectorType::iterator lambdaIt = pcaCovariance.lambdas().begin();
      typename VnlVectorType::iterator lambdaEnd = pcaCovariance.lambdas().end();
      unsigned int                     nonZeroLength = 0;
      for (; lambdaIt != lambdaEnd && (*lambdaIt) > 1e-14; ++lambdaIt, ++nonZeroLength)
      {
      }
      if (this->m_EigenValues != nullptr)
      {
        delete this->m_EigenValues;
      }
      this->m_EigenValues = new VnlVectorType(pcaCovariance.lambdas().extract(nonZeroLength));

      if (this->m_EigenVectors != nullptr)
      {
        delete this->m_EigenVectors;
      }
      this->m_EigenVectors = new VnlMatrixType(pcaCovariance.V().get_n_columns(0, nonZeroLength));

      if (this->m_EigenValuesRegularized == nullptr)
      {
        this->m_EigenValuesRegularized = new vnl_vector<double>(this->m_EigenValues->size());
      }

      vnl_vector<double>::iterator       regularizedValue;
      vnl_vector<double>::const_iterator eigenValue;

      // if there is regularization (>0), the eigenvalues are altered and stored in regularizedValue
      if (this->m_ShrinkageIntensity != 0)
      {
        for (regularizedValue = this->m_EigenValuesRegularized->begin(), eigenValue = this->m_EigenValues->begin();
             regularizedValue != this->m_EigenValuesRegularized->end();
             regularizedValue++, eigenValue++)
        {
          *regularizedValue = -this->m_ShrinkageIntensity * this->m_BaseVariance -
                              this->m_ShrinkageIntensity * this->m_BaseVariance * this->m_ShrinkageIntensity *
                                this->m_BaseVariance / (1.0 - this->m_ShrinkageIntensity) / *eigenValue;
        }
      }
      /** If there is no regularization (m_ShrinkageIntensity==0), a division by zero
       * is avoided by just copying the eigenvalues to regularizedValue.
       * However this will be handled correctly in the calculation of the value and derivative.
       * Providing a non-square eigenvector matrix, with associated eigen values that are
       * non-zero yields a Mahalanobis distance calculation with a pseudo inverse.
       */
      else
      {
        for (regularizedValue = this->m_EigenValuesRegularized->begin(), eigenValue = this->m_EigenValues->begin();
             regularizedValue != this->m_EigenValuesRegularized->end();
             regularizedValue++, eigenValue++)
        {
          *regularizedValue = *eigenValue;
        }
      }
      this->m_InverseCovarianceMatrix = nullptr;
    }
    break;
    case 2: // decomposed scaled covariance (element specific regularization)
    {
      if (this->m_NormalizedShapeModel == false)
      {
        itkExceptionMacro(<< "ShapeModelCalculation option 2 is only implemented for NormalizedShapeModel = true");
      }

      bool pcaNeedsUpdate = false;

      if (this->m_BaseVarianceNeedsUpdate || this->m_VariancesNeedsUpdate)
      {
        pcaNeedsUpdate = true;
        this->m_BaseStd = sqrt(this->m_BaseVariance);
        this->m_CentroidXStd = sqrt(this->m_CentroidXVariance);
        this->m_CentroidYStd = sqrt(this->m_CentroidYVariance);
        this->m_CentroidZStd = sqrt(this->m_CentroidZVariance);
        this->m_SizeStd = sqrt(this->m_SizeVariance);
        vnl_matrix<double> scaledCovariance(*this->m_CovarianceMatrix);

        scaledCovariance.set_columns(0, scaledCovariance.get_n_columns(0, shapeLength) / this->m_BaseStd);
        scaledCovariance.scale_column(shapeLength, 1.0 / this->m_CentroidXStd);
        scaledCovariance.scale_column(shapeLength + 1, 1.0 / this->m_CentroidYStd);
        scaledCovariance.scale_column(shapeLength + 2, 1.0 / this->m_CentroidZStd);
        scaledCovariance.scale_column(shapeLength + 3, 1.0 / this->m_SizeStd);

        scaledCovariance.update(scaledCovariance.get_n_rows(0, shapeLength) / this->m_BaseStd);

        scaledCovariance.scale_row(shapeLength, 1.0 / this->m_CentroidXStd);
        scaledCovariance.scale_row(shapeLength + 1, 1.0 / this->m_CentroidYStd);
        scaledCovariance.scale_row(shapeLength + 2, 1.0 / this->m_CentroidZStd);
        scaledCovariance.scale_row(shapeLength + 3, 1.0 / this->m_SizeStd);

        PCACovarianceType                pcaCovariance(scaledCovariance);
        typename VnlVectorType::iterator lambdaIt = pcaCovariance.lambdas().begin();
        typename VnlVectorType::iterator lambdaEnd = pcaCovariance.lambdas().end();
        unsigned int                     nonZeroLength = 0;
        for (; lambdaIt != lambdaEnd && (*lambdaIt) > 1e-14; ++lambdaIt, ++nonZeroLength)
        {
        }

        if (this->m_EigenValues != nullptr)
        {
          delete this->m_EigenValues;
        }
        this->m_EigenValues = new VnlVectorType(pcaCovariance.lambdas().extract(nonZeroLength));

        if (this->m_EigenVectors != nullptr)
        {
          delete this->m_EigenVectors;
        }
        this->m_EigenVectors = new VnlMatrixType(pcaCovariance.V().get_n_columns(0, nonZeroLength));
      }
      if (this->m_ShrinkageIntensityNeedsUpdate || pcaNeedsUpdate)
      {
        if (this->m_EigenValuesRegularized != nullptr)
        {
          delete this->m_EigenValuesRegularized;
        }
        // if there is regularization (>0), the eigenvalues are altered and kept in regularizedValue
        if (this->m_ShrinkageIntensity != 0)
        {
          this->m_EigenValuesRegularized = new vnl_vector<double>(this->m_EigenValues->size());
          typename vnl_vector<double>::iterator       regularizedValue;
          typename vnl_vector<double>::const_iterator eigenValue;
          for (regularizedValue = this->m_EigenValuesRegularized->begin(), eigenValue = this->m_EigenValues->begin();
               regularizedValue != this->m_EigenValuesRegularized->end();
               regularizedValue++, eigenValue++)
          {
            *regularizedValue = -this->m_ShrinkageIntensity - this->m_ShrinkageIntensity * this->m_ShrinkageIntensity /
                                                                (1.0 - this->m_ShrinkageIntensity) / *eigenValue;
          }
        }
        /** If there is no regularization (m_ShrinkageIntensity==0),
         * a division by zero is avoided by just copying the eigenvalues to regularizedValue.
         * However this will be handled correctly in the calculation of the value and derivative.
         * Providing a non-square eigenvector matrix, with associated eigen values that are
         * non-zero yields a Mahalanobis distance calculation with a pseudo inverse.
         */
        else
        {
          this->m_EigenValuesRegularized = new VnlVectorType(*this->m_EigenValues);
        }
      }
      this->m_ShrinkageIntensityNeedsUpdate = false;
      this->m_BaseVarianceNeedsUpdate = false;
      this->m_VariancesNeedsUpdate = false;
      this->m_InverseCovarianceMatrix = nullptr;
    }
    break;
    default:
      this->m_InverseCovarianceMatrix = nullptr;
      this->m_EigenValuesRegularized = nullptr;
  }

} // end Initialize()


/**
 * ******************* GetValue *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
auto
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::GetValue(
  const TransformParametersType & parameters) const -> MeasureType
{
  /** Sanity checks. */
  FixedPointSetConstPointer fixedPointSet = this->GetFixedPointSet();
  if (!fixedPointSet)
  {
    itkExceptionMacro(<< "Fixed point set has not been assigned");
  }

  /** Initialize some variables */
  // this->m_NumberOfPointsCounted = 0;
  MeasureType value = NumericTraits<MeasureType>::Zero;

  // InputPointType movingPoint;
  OutputPointType fixedPoint;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters(parameters);

  const unsigned int shapeLength = Self::FixedPointSetDimension * (fixedPointSet->GetNumberOfPoints());
  this->m_ProposalVector.set_size(this->m_ProposalLength);

  /** Part 1:
   * - Copy point positions in proposal vector
   */

  /** Create iterators. */
  PointIterator pointItFixed = fixedPointSet->GetPoints()->Begin();
  PointIterator pointEnd = fixedPointSet->GetPoints()->End();

  unsigned int vertexindex = 0;
  /** Loop over the corresponding points. */
  while (pointItFixed != pointEnd)
  {
    fixedPoint = pointItFixed.Value();
    this->FillProposalVector(fixedPoint, vertexindex);

    this->m_NumberOfPointsCounted++;
    ++pointItFixed;
    vertexindex += Self::FixedPointSetDimension;
  } // end loop over all corresponding points

  if (this->m_NormalizedShapeModel)
  {
    /** Part 2:
     * - Calculate shape centroid
     * - put centroid values in proposal
     * - update proposal vector with aligned shape
     */
    this->UpdateCentroidAndAlignProposalVector(shapeLength);

    /** Part 3:
     * - Calculate l2-norm from aligned shapes
     * - put l2-norm value in proposal vector
     * - update proposal vector with size normalized shape
     */
    this->UpdateL2(shapeLength);
    this->NormalizeProposalVector(shapeLength);
  }

  VnlVectorType differenceVector;
  VnlVectorType centerrotated;
  VnlVectorType eigrot;

  this->CalculateValue(value, differenceVector, centerrotated, eigrot);

  return value;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::GetDerivative(const TransformParametersType & parameters,
                                                                             DerivativeType & derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits<MeasureType>::Zero;
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType &                   value,
  DerivativeType &                derivative) const
{
  /** Sanity checks. */
  FixedPointSetConstPointer fixedPointSet = this->GetFixedPointSet();
  if (!fixedPointSet)
  {
    itkExceptionMacro(<< "Fixed point set has not been assigned");
  }

  /** Initialize some variables */
  // this->m_NumberOfPointsCounted = 0;
  value = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits<DerivativeValueType>::ZeroValue());

  // InputPointType movingPoint;
  OutputPointType fixedPoint;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters(parameters);

  const unsigned int shapeLength = Self::FixedPointSetDimension * fixedPointSet->GetNumberOfPoints();

  this->m_ProposalVector.set_size(this->m_ProposalLength);
  this->m_ProposalDerivative = new ProposalDerivativeType(this->GetNumberOfParameters(), nullptr);

  /** Part 1:
   * - Copy point positions in proposal vector
   * - Copy point derivatives in proposal derivative vector
   */

  /** Create iterators. */
  PointIterator pointItFixed = fixedPointSet->GetPoints()->Begin();
  PointIterator pointEnd = fixedPointSet->GetPoints()->End();

  unsigned int vertexindex = 0;
  /** Loop over the corresponding points. */
  while (pointItFixed != pointEnd)
  {
    fixedPoint = pointItFixed.Value();
    this->FillProposalVector(fixedPoint, vertexindex);
    this->FillProposalDerivative(fixedPoint, vertexindex);

    this->m_NumberOfPointsCounted++;
    ++pointItFixed;
    vertexindex += Self::FixedPointSetDimension;
  } // end loop over all corresponding points

  if (this->m_NormalizedShapeModel)
  {
    /** Part 2:
     * - Calculate shape centroid
     * - put centroid values in proposal
     * - update proposal vector with aligned shape
     * - Calculate centroid derivatives and update proposal derivative vectors
     * - put centroid derivatives values in proposal derivatives
     * - update proposal derivatives
     */
    this->UpdateCentroidAndAlignProposalVector(shapeLength);
    this->UpdateCentroidAndAlignProposalDerivative(shapeLength);

    /** Part 3:
     * - Calculate l2-norm from aligned shapes
     * - put l2-norm value in proposal vector
     * - update proposal vector with size normalized shape
     * - Calculate l2-norm derivatice from updated proposal vector
     * - put l2-norm derivative value in proposal derivative vectors
     * - update proposal derivatives
     */
    this->UpdateL2(shapeLength);
    this->UpdateL2AndNormalizeProposalDerivative(shapeLength);
    this->NormalizeProposalVector(shapeLength);

  } // end if(m_NormalizedShapeModel)

  // TODO this declaration instantiates a zero sized vector, but it will be reassigned anyways.
  VnlVectorType differenceVector;
  VnlVectorType centerrotated;
  VnlVectorType eigrot;

  this->CalculateValue(value, differenceVector, centerrotated, eigrot);

  if (value != 0.0)
  {
    this->CalculateDerivative(derivative, value, differenceVector, centerrotated, eigrot, shapeLength);
  }
  else
  {
    typename ProposalDerivativeType::iterator proposalDerivativeIt = this->m_ProposalDerivative->begin();
    typename ProposalDerivativeType::iterator proposalDerivativeEnd = this->m_ProposalDerivative->end();
    for (; proposalDerivativeIt != proposalDerivativeEnd; ++proposalDerivativeIt)
    {
      if (*proposalDerivativeIt != nullptr)
      {
        delete (*proposalDerivativeIt);
      }
    }
  }
  delete this->m_ProposalDerivative;
  this->m_ProposalDerivative = nullptr;

  this->CalculateCutOffValue(value);

} // end GetValueAndDerivative()


/**
 * ******************* FillProposalVector *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::FillProposalVector(const OutputPointType & fixedPoint,
                                                                                  const unsigned int vertexindex) const
{
  OutputPointType mappedPoint;
  /** Get the current corresponding points. */
  mappedPoint = this->m_Transform->TransformPoint(fixedPoint);

  /** Copy n-D coordinates into big Shape vector. Aligning the centroids is done later. */
  for (unsigned int d = 0; d < Self::FixedPointSetDimension; ++d)
  {
    this->m_ProposalVector[vertexindex + d] = mappedPoint[d];
  }
} // end FillProposalVector()


/**
 * ******************* FillProposalDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::FillProposalDerivative(
  const OutputPointType & fixedPoint,
  const unsigned int      vertexindex) const
{
  /**
   * A (column) vector is constructed for each mu, only if that mu affects the shape penalty.
   * I.e. if there is at least one point of the mesh with non-zero derivatives,
   * a full column vector is instantiated (which can contain zeros for many other points)
   *
   * m_ProposalDerivative is a container with either full shape-vector-sized derivative vectors or NULL-s. Example:
   *
   * mu1: [ [ dx1/dmu1 , dy1/dmu1 , dz1/dmu1 ] , [ 0 , 0 , 0 ] , [ dx3/dmu1 , dy3/dmu1 , dz3/dmu1 ] , [...] ]^T
   * mu2: Null
   * mu3: [ [ 0 , 0 , 0 ] , [ dx2/dmu3 , dy2/dmu3 , dz2/dmu3 ] , [ dx3/dmu3 , dy3/dmu3 , dz3/dmu3 ] , [...] ]^T
   *
   */

  NonZeroJacobianIndicesType nzji(this->m_Transform->GetNumberOfNonZeroJacobianIndices());

  /** Get the TransformJacobian dT/dmu. */
  TransformJacobianType jacobian;
  this->m_Transform->GetJacobian(fixedPoint, jacobian, nzji);

  for (unsigned int i = 0; i < nzji.size(); ++i)
  {
    const unsigned int mu = nzji[i];
    if ((*this->m_ProposalDerivative)[mu] == nullptr)
    {
      /** Create the big column vector if it does not yet exist for this mu*/
      (*this->m_ProposalDerivative)[mu] = new VnlVectorType(this->m_ProposalLength, 0.0);
      // memory will be freed in CalculateDerivative()
    }

    /** The column vector exists for this mu, so copy the jacobians for this point into the big vector. */
    for (unsigned int d = 0; d < Self::FixedPointSetDimension; ++d)
    {
      (*(*this->m_ProposalDerivative)[mu])[vertexindex + d] = jacobian.get_column(i)[d];
    }
  }

} // end FillProposalVector()


/**
 * ******************* UpdateCentroidAndAlignProposalVector *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::UpdateCentroidAndAlignProposalVector(
  const unsigned int shapeLength) const
{
  /** Aligning Shapes with their centroids */
  for (unsigned int d = 0; d < Self::FixedPointSetDimension; ++d)
  {
    // Create an alias for the centroid elements in the proposal vector
    double & centroid_d = this->m_ProposalVector[shapeLength + d];

    centroid_d = 0; // initialize centroid x,y,z to zero

    for (unsigned int index = 0; index < shapeLength; index += Self::FixedPointSetDimension)
    {
      // sum all x coordinates to centroid_x, y to centroid_y
      centroid_d += this->m_ProposalVector[index + d];
    }

    // divide sum to get average
    centroid_d /= this->GetFixedPointSet()->GetNumberOfPoints();

    for (unsigned int index = 0; index < shapeLength; index += Self::FixedPointSetDimension)
    {
      // subtract average
      this->m_ProposalVector[index + d] -= centroid_d;
    }
  }

} // end UpdateCentroidAndAlignProposalVector()


/**
 * ******************* UpdateCentroidAndAlignProposalDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::UpdateCentroidAndAlignProposalDerivative(
  const unsigned int shapeLength) const
{
  typename ProposalDerivativeType::iterator proposalDerivativeIt = m_ProposalDerivative->begin();
  typename ProposalDerivativeType::iterator proposalDerivativeEnd = m_ProposalDerivative->end();
  while (proposalDerivativeIt != proposalDerivativeEnd)
  {
    if (*proposalDerivativeIt != nullptr)
    {
      for (unsigned int d = 0; d < Self::FixedPointSetDimension; ++d)
      {
        double & centroid_dDerivative = (**proposalDerivativeIt)[shapeLength + d];
        centroid_dDerivative = 0; // initialize accumulators to zero

        for (unsigned int index = 0; index < shapeLength; index += Self::FixedPointSetDimension)
        {
          centroid_dDerivative += (**proposalDerivativeIt)[index + d]; // sum all x derivatives
        }

        centroid_dDerivative /= this->GetFixedPointSet()->GetNumberOfPoints(); // divide sum to get average

        for (unsigned int index = 0; index < shapeLength; index += Self::FixedPointSetDimension)
        {
          (**proposalDerivativeIt)[index + d] -= centroid_dDerivative; // subtract average
        }
      }
    }
    ++proposalDerivativeIt;
  }
} // end UpdateCentroidAndAlignProposalDerivative()


/**
 * ******************* UpdateL2 *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::UpdateL2(const unsigned int shapeLength) const
{
  double & l2norm = this->m_ProposalVector[shapeLength + Self::FixedPointSetDimension];

  // loop over all shape coordinates of the aligned shape
  l2norm = 0; // initialize l2norm to zero
  for (unsigned int index = 0; index < shapeLength; ++index)
  {
    // accumulate squared distances
    l2norm += this->m_ProposalVector[index] * this->m_ProposalVector[index];
  }
  l2norm = sqrt(l2norm / this->GetFixedPointSet()->GetNumberOfPoints());

} // end UpdateL2AndNormalizeProposalVector()


/**
 * ******************* NormalizeProposalVector *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::NormalizeProposalVector(
  const unsigned int shapeLength) const
{
  double & l2norm = this->m_ProposalVector[shapeLength + Self::FixedPointSetDimension];

  // loop over all shape coordinates of the aligned shape
  for (unsigned int index = 0; index < shapeLength; ++index)
  {
    // normalize shape size by l2-norm
    this->m_ProposalVector[index] /= l2norm;
  }

} // end NormalizeProposalVector()


/**
 * ******************* UpdateL2AndNormalizeProposalDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::UpdateL2AndNormalizeProposalDerivative(
  const unsigned int shapeLength) const
{
  double & l2norm = this->m_ProposalVector[shapeLength + Self::FixedPointSetDimension];

  typename ProposalDerivativeType::iterator proposalDerivativeIt = this->m_ProposalDerivative->begin();
  typename ProposalDerivativeType::iterator proposalDerivativeEnd = this->m_ProposalDerivative->end();

  while (proposalDerivativeIt != proposalDerivativeEnd)
  {
    if (*proposalDerivativeIt != nullptr)
    {
      double & l2normDerivative = (**proposalDerivativeIt)[shapeLength + Self::FixedPointSetDimension];
      l2normDerivative = 0; // initialize to zero
      // loop over all shape coordinates of the aligned shape
      for (unsigned int index = 0; index < shapeLength; ++index)
      {
        l2normDerivative += this->m_ProposalVector[index] * (**proposalDerivativeIt)[index];
      }
      l2normDerivative /= (l2norm * sqrt((double)(this->GetFixedPointSet()->GetNumberOfPoints())));

      // loop over all shape coordinates of the aligned shape
      for (unsigned int index = 0; index < shapeLength; ++index)
      {
        // update normalized shape derivatives
        (**proposalDerivativeIt)[index] = (**proposalDerivativeIt)[index] / l2norm -
                                          this->m_ProposalVector[index] * l2normDerivative / (l2norm * l2norm);
      }
    }
    ++proposalDerivativeIt;
  }

} // end UpdateL2AndNormalizeProposalDerivative()


/**
 * ******************* CalculateValue *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::CalculateValue(MeasureType &   value,
                                                                              VnlVectorType & differenceVector,
                                                                              VnlVectorType & centerrotated,
                                                                              VnlVectorType & eigrot) const
{
  differenceVector = this->m_ProposalVector - *m_MeanVector;

  switch (this->m_ShapeModelCalculation)
  {
    case 0: // full covariance
    {
      value = sqrt(bracket(differenceVector, *this->m_InverseCovarianceMatrix, differenceVector));
      break;
    }
    case 1: // decomposed covariance (uniform regularization)
    {
      centerrotated = differenceVector * (*m_EigenVectors);                /** diff^T * V */
      eigrot = element_quotient(centerrotated, *m_EigenValuesRegularized); /** diff^T * V * Lambda^-1 */
      if (this->m_ShrinkageIntensity != 0)
      {
        /** innerproduct diff^T * V * Lambda^-1 * V^T * diff  +  1/(sigma_0*Beta)* diff^T*diff*/
        value = sqrt(dot_product(eigrot, centerrotated) + dot_product(differenceVector, differenceVector) /
                                                            (this->m_ShrinkageIntensity * this->m_BaseVariance));
      }
      else
      {
        /** innerproduct diff^T * V * Lambda^-1 * V^T * diff*/
        value = sqrt(dot_product(eigrot, centerrotated));
      }
      break;
    }
    case 2: // decomposed scaled covariance (element specific regularization)
    {
      const unsigned int               shapeLength = this->m_ProposalLength - Self::FixedPointSetDimension - 1;
      typename VnlVectorType::iterator diffElementIt = differenceVector.begin();
      for (unsigned int diffElementIndex = 0; diffElementIndex < shapeLength; ++diffElementIndex, ++diffElementIt)
      {
        *diffElementIt /= this->m_BaseStd;
      }
      differenceVector[shapeLength] /= this->m_CentroidXStd;
      differenceVector[shapeLength + 1] /= this->m_CentroidYStd;
      differenceVector[shapeLength + 2] /= this->m_CentroidZStd;
      differenceVector[shapeLength + 3] /= this->m_SizeStd;

      centerrotated = differenceVector * (*this->m_EigenVectors);                /** diff^T * V */
      eigrot = element_quotient(centerrotated, *this->m_EigenValuesRegularized); /** diff^T * V * Lambda^-1 */
      if (this->m_ShrinkageIntensity != 0)
      {
        /** innerproduct diff^T * ~V * I * ~V^T * diff  +  1/(Beta)* diff^T*diff*/
        value =
          sqrt(dot_product(eigrot, centerrotated) + differenceVector.squared_magnitude() / this->m_ShrinkageIntensity);
      }
      else
      {
        /** innerproduct diff^T * V * I * V^T * diff*/
        value = sqrt(dot_product(eigrot, centerrotated));
      }

      break;
    }
    default:
      break;
  }

} // end CalculateValue()


/**
 * ******************* CalculateDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::CalculateDerivative(
  DerivativeType &      derivative,
  const MeasureType &   value,
  const VnlVectorType & differenceVector,
  const VnlVectorType & centerrotated,
  const VnlVectorType & eigrot,
  const unsigned int    shapeLength) const
{
  typename ProposalDerivativeType::iterator proposalDerivativeIt = this->m_ProposalDerivative->begin();
  typename ProposalDerivativeType::iterator proposalDerivativeEnd = this->m_ProposalDerivative->end();

  typename DerivativeType::iterator derivativeIt = derivative.begin();

  for (; proposalDerivativeIt != proposalDerivativeEnd; ++proposalDerivativeIt, ++derivativeIt)
  {
    if (*proposalDerivativeIt != nullptr)
    {
      switch (this->m_ShapeModelCalculation)
      {
        case 0: // full covariance
        {
          /**innerproduct diff^T * Sigma^-1 * d/dmu (diff), where iterated over mu-s*/
          *derivativeIt = bracket(differenceVector, *m_InverseCovarianceMatrix, (**proposalDerivativeIt)) / value;
          this->CalculateCutOffDerivative(*derivativeIt, value);
          break;
        }
        case 1: // decomposed covariance (uniform regularization)
        {
          if (this->m_ShrinkageIntensity != 0)
          {
            /** Innerproduct diff^T * V * Lambda^-1 * V^T * d/dmu(diff)
             * + 1/(Beta*sigma_0^2)*diff^T* d/dmu(diff), where iterated over mu-s
             */
            *derivativeIt = (dot_product(eigrot, this->m_EigenVectors->transpose() * (**proposalDerivativeIt)) +
                             dot_product(differenceVector, **proposalDerivativeIt) /
                               (this->m_ShrinkageIntensity * this->m_BaseVariance)) /
                            value;
            this->CalculateCutOffDerivative(*derivativeIt, value);
          }
          else // m_ShrinkageIntensity==0
          {
            /**innerproduct diff^T * V * Lambda^-1 * V^T * d/dmu (diff), where iterated over mu-s*/
            *derivativeIt = (dot_product(eigrot, this->m_EigenVectors->transpose() * (**proposalDerivativeIt))) / value;
            this->CalculateCutOffDerivative(*derivativeIt, value);
          }
          break;
        }
        case 2: // decomposed scaled covariance (element specific regularization)
        {
          // first scale proposalDerivatives with their sigma's in order to evaluate
          // with the EigenValues and EigenVectors of the scaled CovarianceMatrix
          typename VnlVectorType::iterator propDerivElementIt = (*proposalDerivativeIt)->begin();
          for (unsigned int propDerivElementIndex = 0; propDerivElementIndex < shapeLength;
               ++propDerivElementIndex, ++propDerivElementIt)
          {
            (*propDerivElementIt) /= this->m_BaseStd;
          }
          (**proposalDerivativeIt)[shapeLength] /= this->m_CentroidXStd;
          (**proposalDerivativeIt)[shapeLength + 1] /= this->m_CentroidYStd;
          (**proposalDerivativeIt)[shapeLength + 2] /= this->m_CentroidZStd;
          (**proposalDerivativeIt)[shapeLength + 3] /= this->m_SizeStd;
          if (this->m_ShrinkageIntensity != 0)
          {
            /** innerproduct diff^T * V * Lambda^-1 * V^T * d/dmu(diff)
             * + 1/(Beta*sigma_0^2)*diff^T* d/dmu(diff), where iterated over mu-s
             */
            *derivativeIt = (dot_product(eigrot, this->m_EigenVectors->transpose() * (**proposalDerivativeIt)) +
                             dot_product(differenceVector, **proposalDerivativeIt) / this->m_ShrinkageIntensity) /
                            value;
            this->CalculateCutOffDerivative(*derivativeIt, value);
          }
          else // m_ShrinkageIntensity==0
          {
            /**innerproduct diff^T * V * Lambda^-1 * V^T * d/dmu (diff), where iterated over mu-s*/
            *derivativeIt = (dot_product(eigrot, this->m_EigenVectors->transpose() * (**proposalDerivativeIt))) / value;
            this->CalculateCutOffDerivative(*derivativeIt, value);
          }
          break;
        }
        default:
        {
        }

          delete (*proposalDerivativeIt);
          // nzjacs++;
      }
    }
  }

} // end CalculateDerivative()


/**
 * ******************* CalculateCutOffValue *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::CalculateCutOffValue(MeasureType & value) const
{
  if (this->m_CutOffValue > 0.0)
  {
    value =
      std::log(std::exp(this->m_CutOffSharpness * value) + std::exp(this->m_CutOffSharpness * this->m_CutOffValue)) /
      this->m_CutOffSharpness;
  }
} // end CalculateCutOffValue()


/**
 * ******************* CalculateCutOffDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::CalculateCutOffDerivative(
  typename DerivativeType::element_type & derivativeElement,
  const MeasureType &                     value) const
{
  if (this->m_CutOffValue > 0.0)
  {
    derivativeElement *= 1.0 / (1.0 + std::exp(this->m_CutOffSharpness * (this->m_CutOffValue - value)));
  }
} // end CalculateCutOffDerivative()


/**
 * ******************* PrintSelf *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
StatisticalShapePointPenalty<TFixedPointSet, TMovingPointSet>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  // \todo complete it
  //
  //   if ( this->m_ComputeSquaredDistance )
  //   {
  //     os << indent << "m_ComputeSquaredDistance: True"<< std::endl;
  //   }
  //   else
  //   {
  //     os << indent << "m_ComputeSquaredDistance: False"<< std::endl;
  //   }
} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkStatisticalShapePointPenalty_hxx
