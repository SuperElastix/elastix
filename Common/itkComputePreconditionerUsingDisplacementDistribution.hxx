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
#ifndef itkComputePreconditionerUsingDisplacementDistribution_hxx
#define itkComputePreconditionerUsingDisplacementDistribution_hxx

#include "itkComputePreconditionerUsingDisplacementDistribution.h"

#include <vnl/vnl_math.h>

#include "itkImageScanlineIterator.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkCropImageFilter.h"
#include "itkMirrorPadImageFilter.h"
#include "itkZeroFluxNeumannPadImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"

#include <cmath> // For abs.


namespace itk
{
/**
 * ************************* Constructor ************************
 */

template <class TFixedImage, class TTransform>
ComputePreconditionerUsingDisplacementDistribution<TFixedImage,
                                                   TTransform>::ComputePreconditionerUsingDisplacementDistribution()
{
  this->m_RegularizationKappa = 0.8;
  this->m_MaximumStepLength = 1.0;
  this->m_ConditionNumber = 2.0;
} // end Constructor


/**
 * ************************* Compute ************************
 */

template <class TFixedImage, class TTransform>
void
ComputePreconditionerUsingDisplacementDistribution<TFixedImage, TTransform>::Compute(const ParametersType & mu,
                                                                                     double &               jacg,
                                                                                     double &               maxJJ,
                                                                                     std::string            methods)
{
  itkExceptionMacro(<< "ERROR: do not call");
} // end Compute()


/**
 * ************************* ComputeDistributionTermsUsingSearchDir ************************
 */

template <class TFixedImage, class TTransform>
void
ComputePreconditionerUsingDisplacementDistribution<TFixedImage, TTransform>::ComputeDistributionTermsUsingSearchDir(
  const ParametersType & mu,
  double &               jacg,
  double &               maxJJ,
  std::string            methods)
{
  itkExceptionMacro(<< "ERROR: do not call");
} // end ComputeDistributionTermsUsingSearchDir()


/**
 * ************************* ComputeForBSplineOnly ************************
 */

template <class TFixedImage, class TTransform>
void
ComputePreconditionerUsingDisplacementDistribution<TFixedImage, TTransform>::ComputeForBSplineOnly(
  const ParametersType & mu,
  const double           delta,
  double &               maxJJ,
  ParametersType &       preconditioner)
{
// Select one of the following:
//#define METHOD_BSPLINE 1 // weights method: 1's in middle, 0's on the outside (original code by Yuchuan)
//#define METHOD_BSPLINE 2 // weights method: 1's everywhere
#define METHOD_BSPLINE 3 // weights method: Use the Jacobian

  /** This function computes four terms needed for the automatic parameter
   * estimation using voxel displacement distribution estimation method.
   * The equation number refers to the SPIE paper.
   * Term 1: jacg = mean( J_j * g ) + var( J_j * g ).
   */

  /** Get the number of parameters. */
  const unsigned int numberOfParameters = static_cast<unsigned int>(this->m_Transform->GetNumberOfParameters());

  /** Get the exact gradient. Uses a random coordinate sampler with
   * NumberOfSamplesForPrecondition samples, which equals numberOfParameters.
   */
  DerivativeType exactgradient(numberOfParameters);
  this->GetScaledDerivative(mu, exactgradient);

  /** Get samples. Uses a grid sampler with m_NumberOfJacobianMeasurements samples. */
  ImageSampleContainerPointer sampleContainer;
  this->SampleFixedImageForJacobianTerms(sampleContainer);
  const SizeValueType nrofsamples = sampleContainer->Size();

  /** Get transform and set current position. */
  typename TransformType::Pointer transform = this->m_Transform;
  const unsigned int              outdim = this->m_Transform->GetOutputSpaceDimension();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType sizejacind = this->m_Transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType        jacj(outdim, sizejacind);
  jacj.Fill(0.0);
  NonZeroJacobianIndicesType jacind(sizejacind);

  /**
   * Compute jac * gradient
   */
  DerivativeType jacj_g(outdim);
  jacj_g.Fill(0.0);
  double globalDeformation = 0.0;
  double globalDeformationSquare = 0.0;

  std::vector<double> localStepSizeSquared(numberOfParameters, 0.0);

  /** Create a compact region. */
  // MS: Better cast transform to a B-spline transform and ask for
  // spline order, which implicates support region size
  // MS: why is all this needed?
  // MS: ok, to have 1's in the middle and 0 in outer rim
#if METHOD_BSPLINE == 1
  auto                 compactImage = FixedImageType::New();
  FixedImageRegionType supportRegion;
  FixedImageRegionType compactRegion;

  typename FixedImageRegionType::SizeType size;
  unsigned int                            supportRegionSize = sizejacind / outdim;
  double                                  supportRegionDimension = 1;
  if (outdim == 3)
  {
    supportRegionDimension = std::cbrt(double(supportRegionSize));
  }
  else if (outdim == 2)
  {
    supportRegionDimension = vnl_math::sqr(double(supportRegionSize));
  }
  size.Fill(supportRegionDimension);

  typename FixedImageRegionType::IndexType startIndex;
  startIndex.Fill(0);
  supportRegion.SetSize(size);
  supportRegion.SetIndex(startIndex);

  compactImage->SetRegions(supportRegion);
  compactImage->Allocate();
  compactImage->FillBuffer(0);

  typename FixedImageRegionType::IndexType compactIndex;
  compactIndex.Fill(1);
  SizeValueType compactRegionDimension;
  if (supportRegionDimension == 4)
  {
    compactRegionDimension = 2;
  }
  else
  {
    compactRegionDimension = supportRegionDimension - 1;
  }
  size.Fill(compactRegionDimension);
  compactRegion.SetSize(size);
  compactRegion.SetIndex(compactIndex);

  itk::ImageRegionIterator<FixedImageType> compactImageIterator(compactImage, compactRegion);
  int                                      testIter = 0;
  while (!compactImageIterator.IsAtEnd())
  {
    compactImageIterator.Set(1);
    ++compactImageIterator;
  }
  const FixedImagePixelType * compactSupportVector = compactImage->GetBufferPointer();
#endif

  /** Loop over all voxels in the sample container. */
  ParametersType binCount(numberOfParameters, 0.0);
  unsigned int   samplenr = 0; // needed for global value only

  for (iter = begin; iter != end; ++iter)
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = iter->Value().m_ImageCoordinates;
    this->m_Transform->GetJacobian(point, jacj, jacind);

    /** Compute the product jac_j * gradient. */
    for (unsigned int i = 0; i < outdim; ++i)
    {
      double temp = 0.0;
      for (unsigned int j = 0; j < sizejacind; ++j)
      {
        int pj = jacind[j];
        temp += jacj(i, j) * exactgradient(pj);
      }

      // Use the absolute value
      jacj_g(i) = std::abs(temp);
    }

    /** A support region is where this voxel has the affect on the B-Spline
     * grid mesh, which means each voxel has an influence on multiple grid
     * control point, or means each control point is determined by multiple
     * voxels.
     */
    for (unsigned int j = 0; j < sizejacind; ++j)
    {
      /** Select the only nonzero entry of the displacement jacj_g (B-spline specific). */
      // const unsigned int nonzerodim = j / ( sizejacind / outdim );
      // double displacement = jacj_g[ nonzerodim ];
      // For the affine transform the nonzero dim would be different
      // For a generic transform we would need the norm over all dimensions

      unsigned int nonzerodim = j / outdim; // Affine, first 9 parameters
      if (j >= outdim * outdim)
        nonzerodim = j - outdim * outdim; // Affine, last 3
      if (numberOfParameters > 13)
        nonzerodim = j / (sizejacind / outdim); // B-spline

      double displacement = jacj_g[nonzerodim];

#if METHOD_BSPLINE == 1
      /** Add the deformations on the compact support region. */
      /** Count the numbers of the contributed voxels. */
      if (compactSupportVector[j % supportRegionSize] > 0)
      {
        int pj = jacind[j];
        preconditioner[pj] += displacement;
        localStepSizeSquared[pj] += displacement * displacement;
        binCount[pj] += 1;
      }
      // MS: as far as I understand the above code compactSupportVector[.] will
      // be 1 in the middle and 0 in the outer rim.
#elif METHOD_BSPLINE == 2
      // MS: the following will be all 1 in the complete support region
      int pj = jacind[j];
      preconditioner[pj] += displacement;
      localStepSizeSquared[pj] += displacement * displacement;
      binCount[pj] += 1;
#elif METHOD_BSPLINE == 3
      // MS: the following will use the Jacobian as weights
      const unsigned int pj = jacind[j];
      const double       weight = std::abs(jacj(nonzerodim, j));
      // YQ: the weight is positive.

      /** localStepSize keeps track of the mean displacement.
       * localStepSizeSquared keeps track of the standard deviation.
       */
      preconditioner[pj] += weight * displacement;
      localStepSizeSquared[pj] += weight * displacement * displacement;
      binCount[pj] += weight;
#endif
    }

    /** Add them for global step size. */
    double voxelDeformationTmp = jacj_g.magnitude();
    globalDeformation += voxelDeformationTmp;
    globalDeformationSquare += voxelDeformationTmp * voxelDeformationTmp;

    ++samplenr;
  } // end loop over sample container

  /** Compute the sigma of the distribution of JGG_k.
  double meanGlobalDeformation = globalDeformation / samplenr;
  double global_sigma = globalDeformationSquare - meanGlobalDeformation * globalDeformation;
  global_sigma /= ( nrofsamples - 1 ); // unbiased estimation
  globalStepSize = meanGlobalDeformation + 2.0 * std::sqrt( global_sigma );
  */

  /** Convert the local step sizes to a scaling factor. */
  unsigned int counter_tmp = 0;
  for (unsigned int i = 0; i < numberOfParameters; ++i)
  {
    if (preconditioner[i] > 0)
    {
      /** Mean deformation magnitude. */
      double nonZeroBin = binCount[i];
      if (nonZeroBin > 0)
      {
        const double meanLocalStepSize = preconditioner[i] / nonZeroBin;
        double       sigma = (localStepSizeSquared[i] / nonZeroBin) - meanLocalStepSize * meanLocalStepSize;

        if (sigma < 1e-9)
          sigma = 0.0;

        /** Apply the 2 sigma rule. */
        preconditioner[i] = delta / (std::sqrt(3.0) * (meanLocalStepSize + 2.0 * std::sqrt(sigma)));
      }
    }
    // else this entry remains 0, but this will be fixed later
    else
    {
      ++counter_tmp;
    }

#if 0
    elxout << std::scientific;
    elxout << "The preconditioner before interpolation: [ ";
    //elxout << sigma << " ";
    elxout << preconditioner[i] << " ";
    elxout << "]" << std::endl;
    elxout << std::fixed;
#endif
  } // end loop over localStepSize vector

  if (counter_tmp > 0)
  {
    this->PreconditionerInterpolation(preconditioner);
  }

} // end ComputeForBSplineOnly()


/**
 * ************************* Compute ************************
 */

template <class TFixedImage, class TTransform>
void
ComputePreconditionerUsingDisplacementDistribution<TFixedImage, TTransform>::Compute(const ParametersType & mu,
                                                                                     double &               maxJJ,
                                                                                     ParametersType & preconditioner)
{
  /** Initialize. */
  maxJJ = 0.0;

  /** Get the number of parameters. */
  const unsigned int numberOfParameters = static_cast<unsigned int>(this->m_Transform->GetNumberOfParameters());

  // Replace by a general check later.
  bool transformIsBSpline = false;
  if (numberOfParameters > 13)
    transformIsBSpline = true; // assume B-spline

  /** Get the exact gradient. Uses a random coordinate sampler with
   * NumberOfSamplesForPrecondition samples, which equals numberOfParameters.
   */
  DerivativeType exactgradient(numberOfParameters);
  this->GetScaledDerivative(mu, exactgradient);

  /** Get samples. Uses a grid sampler with m_NumberOfJacobianMeasurements samples. */
  ImageSampleContainerPointer sampleContainer;
  this->SampleFixedImageForJacobianTerms(sampleContainer);
  const SizeValueType nrofsamples = sampleContainer->Size();

  /** Get transform and set current position. */
  typename TransformType::Pointer transform = this->m_Transform;
  const unsigned int              outdim = this->m_Transform->GetOutputSpaceDimension();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType sizejacind = this->m_Transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType        jacj(outdim, sizejacind);
  jacj.Fill(0.0);
  NonZeroJacobianIndicesType jacind(sizejacind);

  /** Declare temporary variables. Not needed for all methods. check later */
  DerivativeType jacj_g(outdim);
  jacj_g.Fill(0.0);
  JacobianType        jacjjacj(outdim, outdim);
  const double        sqrt2 = std::sqrt(static_cast<double>(2.0));
  std::vector<double> localStepSizeSquared(numberOfParameters, 0.0);
  ParametersType      binCount(numberOfParameters);
  binCount.Fill(0.0);

  /** Loop over all voxels in the sample container. */
  for (iter = begin; iter != end; ++iter)
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = iter->Value().m_ImageCoordinates;
    this->m_Transform->GetJacobian(point, jacj, jacind);

    /** Compute 1st part of JJ: ||J_j||_F^2. */
    double JJ_j = vnl_math::sqr(jacj.frobenius_norm());

    /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F. */
    vnl_fastops::ABt(jacjjacj, jacj, jacj);
    JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

    /** Max_j [JJ_j]. */
    maxJJ = std::max(maxJJ, JJ_j);

    double displacement2_j = 0.0;
    if (transformIsBSpline)
    {
      for (unsigned int i = 0; i < outdim; ++i)
      {
        double temp = 0.0;
        for (unsigned int j = 0; j < sizejacind; ++j)
        {
          int pj = jacind[j];
          temp += jacj(i, j) * exactgradient(pj);
        }

        // Use the absolute value
        jacj_g(i) = std::abs(temp);
      }
      displacement2_j = jacj_g.magnitude();
    }

    /** Update all entries of the pre-conditioner. */
    for (unsigned int j = 0; j < sizejacind; ++j)
    {
      const unsigned int pj = jacind[j];
      double             displacement_j = 0.0;
      double             jacj_current = 0.0;
      for (unsigned int i = 0; i < outdim; ++i)
      {
        jacj_current += std::abs(jacj(i, j));
      }
      displacement_j = std::abs(jacj_current * exactgradient(pj));

      if (transformIsBSpline)
      {
        displacement_j =
          displacement_j * this->m_RegularizationKappa + (1.0 - this->m_RegularizationKappa) * displacement2_j;
      }
      else
      { // else for affine and rigid
        double diff_jacobian = 0;
        double weight = 0;
        double sum_displacement = 0;
        double sum_weight = 0;
        double weight_sigma = 0.01;
        double maxdiff = 0.0;
        double mindiff = 0.0;
        bool   mindiffCheck = true;

        /** Obtain the maximum and minimum difference of absolute jacobian. */
        for (unsigned int k = 0; k < sizejacind; ++k)
        {
          if (k != j)
          {
            double jacj_k = 0.0;
            for (unsigned int i = 0; i < outdim; ++i)
            {
              jacj_k += std::abs(jacj(i, k));
            }
            diff_jacobian = std::abs(jacj_k - jacj_current);
            if (diff_jacobian > 0 && mindiffCheck)
            {
              mindiff = diff_jacobian;
              mindiffCheck = false;
            }
            if (diff_jacobian > 0 && !mindiffCheck)
            {
              mindiff = diff_jacobian < mindiff ? diff_jacobian : mindiff;
            }
            maxdiff = diff_jacobian > maxdiff ? diff_jacobian : maxdiff;
          } // end if
        }   // end for

        if (maxdiff > 0)
        {
          weight_sigma = mindiff / maxdiff;
        }
        else
        {
          weight_sigma = 1e-9;
        }

        /** To regularize the other entries using the neighborhood information. */
        for (unsigned int k = 0; k < sizejacind; ++k)
        {
          const unsigned int pk = jacind[k];
          if (k != j)
          {
            double jacj_k = 0.0;
            for (unsigned int i = 0; i < outdim; ++i)
            {
              jacj_k += std::abs(jacj(i, k));
            }

            diff_jacobian = std::abs(jacj_k - jacj_current);
            weight = std::exp(-(vnl_math::sqr(diff_jacobian / weight_sigma) / 2.0));

            sum_displacement += std::abs(jacj_k * exactgradient(pk)) * weight;
            sum_weight += weight;
          } // end if
        }   // end for loop regularization

        if (sum_weight > 0.0)
        {
          sum_displacement /= sum_weight;

          /** regularize. */
          displacement_j =
            displacement_j * this->m_RegularizationKappa + (1.0 - this->m_RegularizationKappa) * sum_displacement;
        }
      } // end else for affine and rigid

      /** Compute the displacement due to a change in this parameter. */
      /** localStepSize keeps track of the mean displacement.
       * localStepSizeSquared keeps track of the standard deviation.
       */
      preconditioner[pj] += displacement_j;
      localStepSizeSquared[pj] += displacement_j * displacement_j;
      binCount[pj] += 1.0;
    }
  } // end loop over sample container


  /** Compute the mean local step sizes and apply the 2 sigma rule. */
  double maxEigenvalue = -1e+9;
  double minEigenvalue = 1e+9;
  for (unsigned int i = 0; i < numberOfParameters; ++i)
  {
    /** Mean deformation magnitude. */
    double nonZeroBin = binCount[i];

    const double meanLocalStepSize = preconditioner[i] / (nonZeroBin + 1e-14);
    double       sigma = localStepSizeSquared[i] / (nonZeroBin + 1e-14) - meanLocalStepSize * meanLocalStepSize;

    /** Due to numerical issues, in case of very small squared sums and means,
     * the standard deviation may become negative. This happens for example in
     * case of an affine transformation for the translational parameters.
     */
    if (sigma < 1e-14)
      sigma = 0;

    /** Apply the 2 sigma rule. */
    double localStep = meanLocalStepSize + 2.0 * std::sqrt(sigma) + 1e-14;

    minEigenvalue = std::min(localStep, minEigenvalue);
    maxEigenvalue = std::max(localStep, maxEigenvalue);
    preconditioner[i] = this->m_MaximumStepLength / localStep;

  } // end loop over step size vector

  /** Constrained the condition number into a given range, here we first try kappa = 2. */
  double conditionNumber = maxEigenvalue / minEigenvalue;

#if 1
  elxout << std::scientific;
  elxout << "The max eigen value is: [ ";
  elxout << maxEigenvalue << " ";
  elxout << "]" << std::endl;
  elxout << "The min eigen value is: [ ";
  elxout << minEigenvalue << " ";
  elxout << "]" << std::endl;
  elxout << "The condition number before constraints is: [ ";
  elxout << conditionNumber << " ";
  elxout << "]" << std::endl;
  elxout << std::fixed;
#endif

  if (transformIsBSpline && conditionNumber > this->m_ConditionNumber)
  {
    minEigenvalue = maxEigenvalue / this->m_ConditionNumber;
    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      if (preconditioner[i] > this->m_MaximumStepLength / minEigenvalue)
      {
        preconditioner[i] = this->m_MaximumStepLength / minEigenvalue;
      }
    }
  } // end condition number check.

} // end Compute()


/**
 * ************************* ComputeJacobiTypePreconditioner ************************
 */

template <class TFixedImage, class TTransform>
void
ComputePreconditionerUsingDisplacementDistribution<TFixedImage, TTransform>::ComputeJacobiTypePreconditioner(
  const ParametersType & mu,
  double &               maxJJ,
  ParametersType &       preconditioner)
{
  /** Initialize. */
  maxJJ = 0.0;

  /** Get the number of parameters. */
  const unsigned int numberOfParameters = static_cast<unsigned int>(this->m_Transform->GetNumberOfParameters());

  // Replace by a general check later.
  bool transformIsBSpline = false;
  if (numberOfParameters > 13)
    transformIsBSpline = true; // assume B-spline

  /** Get samples. Uses a grid sampler with m_NumberOfJacobianMeasurements samples. */
  ImageSampleContainerPointer sampleContainer;
  this->SampleFixedImageForJacobianTerms(sampleContainer);
  const SizeValueType nrofsamples = sampleContainer->Size();

  /** Get transform and set current position. */
  typename TransformType::Pointer transform = this->m_Transform;
  const unsigned int              outdim = this->m_Transform->GetOutputSpaceDimension();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType sizejacind = this->m_Transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType        jacj(outdim, sizejacind);
  jacj.Fill(0.0);
  JacobianType               jacjjacj(outdim, outdim);
  const double               sqrt2 = std::sqrt(static_cast<double>(2.0));
  NonZeroJacobianIndicesType jacind(sizejacind);
  ParametersType             binCount(numberOfParameters, 0.0);

  /** Loop over all voxels in the sample container. */
  for (iter = begin; iter != end; ++iter)
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = iter->Value().m_ImageCoordinates;
    this->m_Transform->GetJacobian(point, jacj, jacind);

    /** Compute 1st part of JJ: ||J_j||_F^2. */
    double JJ_j = vnl_math::sqr(jacj.frobenius_norm());

    /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F. */
    vnl_fastops::ABt(jacjjacj, jacj, jacj);
    JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

    /** Max_j [JJ_j]. */
    maxJJ = std::max(maxJJ, JJ_j);

    for (unsigned int i = 0; i < outdim; ++i)
    {
      for (unsigned int j = 0; j < sizejacind; ++j)
      {
        const unsigned int pj = jacind[j];
        preconditioner[pj] += vnl_math::sqr(jacj(i, j));
        binCount[pj] += 1;
      }
    }
  }

  double maxEigenvalue = -1e+9;
  double minEigenvalue = 1e+9;
  for (unsigned int i = 0; i < numberOfParameters; ++i)
  {
    double nonZeroBin = binCount[i] / outdim;
    if (nonZeroBin > 0 && preconditioner[i] > 1e-9)
    {
      double eigenvalue = std::sqrt(preconditioner[i] / (nonZeroBin)) + 1e-14;
      maxEigenvalue = std::max(eigenvalue, maxEigenvalue);
      minEigenvalue = std::min(eigenvalue, minEigenvalue);
      preconditioner[i] = 1.0 / eigenvalue;
    }
  }

#if 0
  elxout << std::scientific;
  elxout << "The max eigen value is: [ ";
  elxout << maxEigenvalue << " ";
  elxout << "]" << std::endl;
  elxout << "The min eigen value is: [ ";
  elxout << minEigenvalue << " ";
  elxout << "]" << std::endl;
#endif

  /** Condition number check. */
  double conditionNumber = maxEigenvalue / minEigenvalue;

  if (transformIsBSpline && conditionNumber > this->m_ConditionNumber)
  {
    minEigenvalue = maxEigenvalue / this->m_ConditionNumber;
    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      if (preconditioner[i] > 1.0 / minEigenvalue)
      {
        preconditioner[i] = 1.0 / minEigenvalue;
      }
    }
  }

#if 0
  elxout << std::scientific;
  elxout << "The condition number after constraints is: [ ";
  elxout << maxEigenvalue / minEigenvalue << " ";
  elxout << "]" << std::endl;
  elxout << std::fixed;
#endif
} // end ComputeJacobiTypePreconditioner()


/**
 * ************************* PreconditionerInterpolation ************************
 */

template <class TFixedImage, class TTransform>
void
ComputePreconditionerUsingDisplacementDistribution<TFixedImage, TTransform>::PreconditionerInterpolation(
  ParametersType & preconditioner)
{
  // Note: This function is only meant for the B-spline transformation
#define UseOldMethod
  const unsigned int SplineOrder = 3;
  using CombinationTransformType = AdvancedCombinationTransform<double, FixedImageDimension>;
  using BSplineTransformType = AdvancedBSplineDeformableTransform<double, FixedImageDimension, SplineOrder>;
  using GridSizeType = typename BSplineTransformType::SizeType;
  using GridIndexType = typename BSplineTransformType::IndexType;
  using GridSpacingType = typename BSplineTransformType::SpacingType;
  using GridOriginType = typename BSplineTransformType::OriginType;
  using GridDirectionType = typename BSplineTransformType::DirectionType;
  using GridRegionType = typename BSplineTransformType::RegionType;
  using CoefficientImageType = typename BSplineTransformType::ImageType;

  using IteratorType = ImageRegionIteratorWithIndex<CoefficientImageType>;
  using ImageScanlineIteratorType = ImageLinearIteratorWithIndex<CoefficientImageType>;
  using SliceIteratorType = ImageSliceIteratorWithIndex<CoefficientImageType>;

  using CropImageFilterType = CropImageFilter<CoefficientImageType, CoefficientImageType>;
  // typedef MirrorPadImageFilter<CoefficientImageType,CoefficientImageType> PadImageFilterType;
  using PadImageFilterType = ZeroFluxNeumannPadImageFilter<CoefficientImageType, CoefficientImageType>;
  using SmoothingFilterType = SmoothingRecursiveGaussianImageFilter<CoefficientImageType, CoefficientImageType>;

  CombinationTransformType * testPtr_combo = dynamic_cast<CombinationTransformType *>(this->m_Transform.GetPointer());
  if (!testPtr_combo)
    return; // throw an error?
  const auto testPtr_bspline = dynamic_cast<const BSplineTransformType *>(testPtr_combo->GetCurrentTransform());
  if (!testPtr_bspline)
    return; // throw an error?

  GridRegionType    gridRegion = testPtr_bspline->GetGridRegion();
  GridSpacingType   gridSpacing = testPtr_bspline->GetGridSpacing();
  GridOriginType    gridOrigin = testPtr_bspline->GetGridOrigin();
  GridDirectionType gridDirection = testPtr_bspline->GetGridDirection();

  auto coefImage = CoefficientImageType::New();
  coefImage->SetRegions(gridRegion);
  coefImage->SetSpacing(gridSpacing);
  coefImage->SetOrigin(gridOrigin);
  coefImage->SetDirection(gridDirection);
  coefImage->Allocate();

  //   auto mask = MaskImageType::New();
  //   mask->CopyInformation( coefImage );
  //   mask->Allocate();
  //   mask->FillBuffer( 0 );

  GridRegionType region2 = gridRegion;
  GridSizeType & size2 = region2.GetModifiableSize();
  size2[0] = size2[0] - SplineOrder;
  size2[1] = size2[1] - SplineOrder;
  size2[2] = size2[2] - SplineOrder;
  GridIndexType & index2 = region2.GetModifiableIndex();
  index2[0] = SplineOrder - 1;
  index2[1] = SplineOrder - 1;
  index2[2] = SplineOrder - 1;

  // Loop over the x, y and z parts of the B-spline separately.
  for (unsigned int i = 0; i < FixedImageDimension; ++i)
  {
    // Copy the preconditioner values to an image
    IteratorType it(coefImage, coefImage->GetLargestPossibleRegion());
    unsigned int k = i * gridRegion.GetNumberOfPixels();
    while (!it.IsAtEnd())
    {
#ifdef UseOldMethod // copy only inner region
      // std::cerr << preconditioner[ i ] << " ";
      if (region2.IsInside(it.GetIndex()))
      {
        it.Set(preconditioner[k]);
      }
      else
      {
        it.Set(-1.0);
      }
#else // copy all, todo: can we trust the outer rim?
      it.Set(preconditioner[k]);
#endif

      ++k;
      ++it;
    }

    // tmp write
    //     using WriterType = ImageFileWriter<CoefficientImageType>;
    //     auto writer1 = WriterType::New();
    //     writer1->SetFileName( "P_0.mha" );
    //     writer1->SetInput( coefImage );
    //     writer1->Update();

    // first time smooth
    auto smoother = SmoothingFilterType::New();
    //     smoother->SetInput(coefImage);
    //     smoother->SetSigma(0.5);
    //     smoother->Update();
    //
    //     // tmp write
    //     auto writer3 = WriterType::New();
    //     writer3->SetFileName("P_coefImageSmooth.mha");
    //     //writer2->SetInput( padder->GetOutput() );
    //     writer3->SetInput(smoother->GetOutput());
    //     writer3->Update();

#ifdef UseOldMethod
    // Fill the holes by a left-right sweep on the interior only, i.e. not
    // considering the part outside the image.
    // ImageScanlineIteratorType it2 = ImageScanlineIteratorType( coefImage, coefImage->GetLargestPossibleRegion() );
    ImageScanlineIteratorType it2 = ImageScanlineIteratorType(coefImage, region2);
    while (!it2.IsAtEnd())
    {
      // forward
      double previous = -1.0;
      while (!it2.IsAtEndOfLine())
      {
        double current = it2.Value();
        //        if( (current == 0.0 && previous > 0.0) || ( current > upperBound && previous < upperBound ))
        if (current == 0.0 && previous > 0.0)
        {
          it2.Set(previous);
        }
        else
        {
          previous = current;
        }
        ++it2;
      }

      // backward
      it2.GoToEndOfLine();
      previous = -1.0;
      while (!it2.IsAtReverseEndOfLine())
      {
        double current = it2.Value();
        if (current == 0.0 && previous > 0.0)
        {
          it2.Set(previous);
        }
        else
        {
          previous = current;
        }
        --it2;
      }

      it2.NextLine();
    }

    SliceIteratorType itSlice = SliceIteratorType(coefImage, region2);
    itSlice.SetFirstDirection(2);
    itSlice.SetSecondDirection(0);
    while (!itSlice.IsAtEnd())
    {
      while (!itSlice.IsAtEndOfSlice())
      {
        // forward
        double previous = -1.0;
        while (!itSlice.IsAtEndOfLine())
        {
          double current = itSlice.Value();
          if (current == 0.0 && previous > 0.0)
          {
            itSlice.Set(previous);
          }
          else
          {
            previous = current;
          }
          ++itSlice;
        }

        itSlice.NextLine();
      }
      itSlice.NextSlice();
    }

    itSlice.GoToReverseBegin();
    while (!itSlice.IsAtReverseEnd())
    {
      while (!itSlice.IsAtReverseEndOfSlice())
      {
        // backward
        double previous = -1.0;
        while (!itSlice.IsAtReverseEndOfLine())
        {
          double current = itSlice.Value();
          if (current == 0.0 && previous > 0.0)
          {
            itSlice.Set(previous);
          }
          else
          {
            previous = current;
          }
          --itSlice;
        }
        itSlice.PreviousLine();
      }
      itSlice.PreviousSlice();
    }
#else // use new method
    // average over the neighborhood using only the non -1 entries
    // we should repeat this a couple of times, until no -1's are left
    double tmp = 0.0;
    GridSizeType radius;
    radius.Fill(1);
    NeighborhoodIterator<CoefficientImageType> nit(radius, coefImage, region2);
    while (!nit.IsAtEnd())
    {
      if (nit.GetCenterPixel() > -0.5)
      {
        ++nit;
        continue;
      }

      // average over the neighborhood
      double accum = 0.0;
      unsigned int count = 0;
      for (unsigned int i = 0; i < nit.Size(); ++i)
      {
        tmp = nit.GetPixel(i);
        if (tmp > -0.5)
        {
          accum += tmp;
          ++count;
        }
      }
      if (count > 0)
      {
        nit.SetCenterPixel(accum / static_cast<double>(count));
      }
      ++nit;
    }
#endif

    // tmp write
    //     auto writer2 = WriterType::New();
    //     writer2->SetFileName("P_1.mha");
    //     writer2->SetInput( coefImage );
    //     writer2->Update();

#ifdef UseOldMethod
    GridSizeType size_tmp;
    // First remove the outer border with -1 information
    auto cropper = CropImageFilterType::New();
    cropper->SetInput(coefImage);
    size_tmp.Fill(1);
    cropper->SetUpperBoundaryCropSize(size_tmp);
    size_tmp.Fill(2);
    cropper->SetLowerBoundaryCropSize(size_tmp);

    // Then add the border using zero flux Neumann
    auto padder = PadImageFilterType::New();
    padder->SetInput(cropper->GetOutput());
    size_tmp.Fill(1);
    padder->SetPadUpperBound(size_tmp);
    size_tmp.Fill(2);
    padder->SetPadLowerBound(size_tmp);
#endif

    // smooth?
    // auto smoother = SmoothingFilterType::New();
#ifdef UseOldMethod
    smoother->SetInput(padder->GetOutput());
#else
    smoother->SetInput(coefImage);
#endif
    smoother->SetSigma(0.5);

    // tmp write
    // auto writer2 = WriterType::New();
    // writer2->SetFileName( "P_3.mha" );
#ifdef UseOldMethod
    // writer2->SetInput( padder->GetOutput() );
    // writer2->SetInput( smoother->GetOutput() );
#else
    writer2->SetInput(coefImage);
    // writer2->SetInput( smoother->GetOutput() );
#endif
    // writer2->Update();

    // Copy back
    k = i * gridRegion.GetNumberOfPixels();
#ifdef UseOldMethod
    IteratorType it3(padder->GetOutput(), padder->GetOutput()->GetLargestPossibleRegion());
    // IteratorType it3( smoother->GetOutput(), smoother->GetOutput()->GetLargestPossibleRegion() );
#else
    IteratorType it3(coefImage, coefImage->GetLargestPossibleRegion());
    // IteratorType it3( smoother->GetOutput(), smoother->GetOutput()->GetLargestPossibleRegion() );
#endif
    while (!it3.IsAtEnd())
    {
      preconditioner[k] = it3.Value() + 1e-8;
      ++k;
      ++it3;
    }
  } // end loop over all dimensions

  //   unsigned int P = localStepSize.size();
  //   for( unsigned int i = 0; i < P; ++i )
  //   {
  //     std::cerr << localStepSize[ i ] << " ";
  //   }
  //   std::cerr << std::endl;

} // end PreconditionerInterpolation()


} // end namespace itk

#endif // end #ifndef itkComputePreconditionerUsingDisplacementDistribution_hxx
