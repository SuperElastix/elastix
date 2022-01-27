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

#ifndef itkParzenWindowNormalizedMutualInformationImageToImageMetric_hxx
#define itkParzenWindowNormalizedMutualInformationImageToImageMetric_hxx

#include "itkParzenWindowNormalizedMutualInformationImageToImageMetric.h"

#include "itkImageLinearConstIteratorWithIndex.h"
#include <vnl/vnl_math.h>

namespace itk
{

/**
 * ********************* PrintSelf ******************************
 *
 * Print out internal information about this class.
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowNormalizedMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os,
                                                                                                Indent indent) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf(os, indent);

  /** This function is not complete, but we don't use it anyway. */

} // end PrintSelf()


/**
 * ********************** ComputeLogMarginalPDF***********************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowNormalizedMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::ComputeLogMarginalPDF(
  MarginalPDFType & pdf) const
{
  /** Typedef iterator */
  using MarginalPDFIteratorType = typename MarginalPDFType::iterator;

  /** Prepare iterators for computing marginal logPDF. */
  MarginalPDFIteratorType       PDFit = pdf.begin();
  const MarginalPDFIteratorType PDFend = pdf.end();

  /** do it! */
  while (PDFit != PDFend)
  {
    if ((*PDFit) > 1e-16)
    {
      (*PDFit) = std::log(*PDFit);
    }
    else
    {
      (*PDFit) = 0.0;
    }
    ++PDFit;
  }

} // end ComputeLogMarginalPDF


/**
 * ********************** ComputeNormalizedMutualInformation ***********************
 * Assumes the marginal pdfs are already log'ed
 * Returns the normalized mutual information, so not its negative...
 */

template <class TFixedImage, class TMovingImage>
auto
ParzenWindowNormalizedMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::
  ComputeNormalizedMutualInformation(MeasureType & jointEntropy) const -> MeasureType
{
  /** Typedef iterators */
  using JointPDFConstIteratorType = ImageLinearConstIteratorWithIndex<JointPDFType>;
  using MarginalPDFConstIteratorType = typename MarginalPDFType::const_iterator;

  /** Prepare iterators for computing measure */
  JointPDFConstIteratorType jointPDFconstit(this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion());
  jointPDFconstit.SetDirection(0);
  jointPDFconstit.GoToBegin();
  MarginalPDFConstIteratorType       fixedPDFconstit = this->m_FixedImageMarginalPDF.begin();
  MarginalPDFConstIteratorType       movingPDFconstit = this->m_MovingImageMarginalPDF.begin();
  const MarginalPDFConstIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
  const MarginalPDFConstIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();

  /** Loop over histogram to compute measure */
  double sumnum = 0.0;
  double sumden = 0.0;
  while (fixedPDFconstit != fixedPDFend)
  {
    const double logFixedImagePDFValue = *fixedPDFconstit;
    movingPDFconstit = this->m_MovingImageMarginalPDF.begin();
    while (movingPDFconstit != movingPDFend)
    {
      const double logMovingImagePDFValue = *movingPDFconstit;
      const double jointPDFValue = jointPDFconstit.Get();
      sumnum -= jointPDFValue * (logFixedImagePDFValue + logMovingImagePDFValue);
      /** check for non-zero bin contribution */
      if (jointPDFValue > 1e-16)
      {
        sumden -= jointPDFValue * std::log(jointPDFValue);
      }
      ++movingPDFconstit;
      ++jointPDFconstit;
    } // end while-loop over moving index
    ++fixedPDFconstit;
    jointPDFconstit.NextLine();
  } // end while-loop over fixed index

  jointEntropy = sumden;
  return static_cast<MeasureType>(sumnum / sumden);
} // end ComputeNormalizedMutualInformation


/**
 * ************************** GetValue **************************
 * Get the match Measure.
 */

template <class TFixedImage, class TMovingImage>
auto
ParzenWindowNormalizedMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::GetValue(
  const ParametersType & parameters) const -> MeasureType
{
  /** Construct the JointPDF and Alpha */
  this->ComputePDFs(parameters);

  /** Normalize the pdfs: p = alpha h */
  this->NormalizeJointPDF(this->m_JointPDF, this->m_Alpha);

  /** Compute the fixed and moving marginal pdfs, by summing over the joint pdf */
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_FixedImageMarginalPDF, 0);
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_MovingImageMarginalPDF, 1);

  /** Replace the probabilities by log(probabilities) */
  this->ComputeLogMarginalPDF(this->m_FixedImageMarginalPDF);
  this->ComputeLogMarginalPDF(this->m_MovingImageMarginalPDF);

  /** Compute the measure */
  MeasureType       jointEntropy = 0.0;
  const MeasureType nMI = this->ComputeNormalizedMutualInformation(jointEntropy);

  return static_cast<MeasureType>(-1.0 * nMI);

} // end GetValue


/**
 * ******************** GetValueAndDerivative *******************
 * Get both the Value and the Derivative of the Measure.
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowNormalizedMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  /** Initialize some variables */
  value = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(0.0);

  /** Construct the JointPDF, JointPDFDerivatives, and Alpha. */
  this->ComputePDFsAndPDFDerivatives(parameters);

  /** Normalize the pdfs: p = alpha h*/
  this->NormalizeJointPDF(this->m_JointPDF, this->m_Alpha);

  /** Compute the fixed and moving marginal pdf by summing over the histogram */
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_FixedImageMarginalPDF, 0);
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_MovingImageMarginalPDF, 1);

  /** Replace the probabilities by log(probabilities) */
  this->ComputeLogMarginalPDF(this->m_FixedImageMarginalPDF);
  this->ComputeLogMarginalPDF(this->m_MovingImageMarginalPDF);

  /** Compute the measure and joint entropy (which we both need to compute the derivative) */
  MeasureType       jointEntropy = 0.0;
  const MeasureType nMI = this->ComputeNormalizedMutualInformation(jointEntropy);
  value = static_cast<MeasureType>(-1.0 * nMI);

  /** Now compute the derivatives:
   * -dNMI/dmu = - 1/Ej [
   *      sum_k sum_i dpdmu(i,k) ( NMI log(p(i,k)) - log(pf(k)) - log(pm(i)) ) ]
   *           = - 1/Ej [ sum_k sum_i dpdmu(i,k) pRatio ]
   * where:
   * dpdmu(i,k) = alpha dhdmu(i,k)
   *
   * m_JointPDFDerivatives reflects dhdmu(i,k) at this point in the code.
   * p = m_JointPDF reflects [alpha h(i,k)]
   *
   * So, we can write, following more or less the source code below:
   * -dNMI/dmu = - sum_k sum_i dhdmu(i,k) alpha*pRatio/Ej
   **/

  /** Typedefs for iterators */
  using JointPDFDerivativesConstIteratorType = ImageLinearConstIteratorWithIndex<JointPDFDerivativesType>;
  using DerivativeIteratorType = typename DerivativeType::iterator;
  using DerivativeConstIteratorType = typename DerivativeType::const_iterator;
  using JointPDFConstIteratorType = ImageLinearConstIteratorWithIndex<JointPDFType>;
  using MarginalPDFConstIteratorType = typename MarginalPDFType::const_iterator;

  /** Setup iterators */
  JointPDFDerivativesConstIteratorType jointPDFDerivativesConstit(
    this->m_JointPDFDerivatives, this->m_JointPDFDerivatives->GetLargestPossibleRegion());
  jointPDFDerivativesConstit.SetDirection(0);
  jointPDFDerivativesConstit.GoToBegin();

  JointPDFConstIteratorType jointPDFconstit(this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion());
  jointPDFconstit.SetDirection(0);
  jointPDFconstit.GoToBegin();

  DerivativeIteratorType            derivit = derivative.begin();
  const DerivativeConstIteratorType derivend = derivative.end();

  MarginalPDFConstIteratorType       fixedPDFconstit = this->m_FixedImageMarginalPDF.begin();
  MarginalPDFConstIteratorType       movingPDFconstit = this->m_MovingImageMarginalPDF.begin();
  const MarginalPDFConstIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
  const MarginalPDFConstIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();

  /** Compute the derivatives */
  while (fixedPDFconstit != fixedPDFend)
  {
    const double logFixedImagePDFValue = *fixedPDFconstit;
    movingPDFconstit = this->m_MovingImageMarginalPDF.begin();
    while (movingPDFconstit != movingPDFend)
    {
      const double logMovingImagePDFValue = *movingPDFconstit;
      const double jointPDFValue = jointPDFconstit.Get();
      if (jointPDFValue > 1e-16)
      {
        const double pRatio =
          (nMI * std::log(jointPDFValue) - logFixedImagePDFValue - logMovingImagePDFValue) / jointEntropy;
        const double pRatioAlpha = this->m_Alpha * pRatio;
        /** check for non-zero bin contribution */
        derivit = derivative.begin();
        while (derivit != derivend)
        {
          (*derivit) -= jointPDFDerivativesConstit.Get() * pRatioAlpha;
          ++derivit;
          ++jointPDFDerivativesConstit;
        } // end while-loop over parameters
      }   // end if-block to check non-zero bin contribution
      ++movingPDFconstit;
      ++jointPDFconstit;
      jointPDFDerivativesConstit.NextLine();
    } // end while-loop over moving index
    ++fixedPDFconstit;
    jointPDFconstit.NextLine();
  } // end while-loop over fixed index

} // end GetValueAndDerivative


} // end namespace itk

#endif // end #ifndef itkParzenWindowNormalizedMutualInformationImageToImageMetric_hxx
