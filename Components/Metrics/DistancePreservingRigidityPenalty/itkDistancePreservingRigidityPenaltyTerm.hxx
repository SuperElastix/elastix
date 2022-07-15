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
#ifndef itkDistancePreservingRigidityPenaltyTerm_hxx
#define itkDistancePreservingRigidityPenaltyTerm_hxx

#include "itkDistancePreservingRigidityPenaltyTerm.h"

#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "itkImageRegionIterator.h"

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template <class TFixedImage, class TScalarType>
DistancePreservingRigidityPenaltyTerm<TFixedImage, TScalarType>::DistancePreservingRigidityPenaltyTerm()
{
  /** Values. */
  this->m_RigidityPenaltyTermValue = NumericTraits<MeasureType>::Zero;

  /** Images required for penalty calculation */
  this->m_BSplineKnotImage = nullptr;
  this->m_PenaltyGridImage = nullptr;
  this->m_SegmentedImage = nullptr;
  this->m_SampledSegmentedImage = nullptr;

  /** Number of the penalty grid points, which belong to rigid regions */
  this->m_NumberOfRigidGrids = 0;

  /** We don't use an image sampler for this advanced metric. */
  this->SetUseImageSampler(false);

} // end Constructor


/**
 * *********************** Initialize *****************************
 */
template <class TFixedImage, class TScalarType>
void
DistancePreservingRigidityPenaltyTerm<TFixedImage, TScalarType>::Initialize()
{
  /** Call the initialize of the superclass. */
  this->Superclass::Initialize();

  /** Check if this transform is a B-spline transform. */
  typename BSplineTransformType::Pointer localBSplineTransform; // default-constructed (null)
  bool                                   transformIsBSpline = this->CheckForBSplineTransform2(localBSplineTransform);
  if (transformIsBSpline)
  {
    this->SetBSplineTransform(localBSplineTransform);
  }

  /** Set the B-spline transform to m_RigidityPenaltyTermMetric. */
  if (!transformIsBSpline)
  {
    itkExceptionMacro(<< "ERROR: this metric expects a B-spline transform.");
  }

  /** Initialize BSplineKnotImage. */
  this->m_BSplineKnotImage = BSplineKnotImageType::New();

  typename TransformType::ParametersType fixedParameters = this->m_Transform->GetFixedParameters();

  typename FixedImageType::SizeType bSplineKnotSize;
  bSplineKnotSize[0] = static_cast<unsigned int>(fixedParameters[0]);
  bSplineKnotSize[1] = static_cast<unsigned int>(fixedParameters[1]);
  bSplineKnotSize[2] = static_cast<unsigned int>(fixedParameters[2]);

  typename FixedImageType::PointType bSplineKnotOrigin;
  bSplineKnotOrigin[0] = fixedParameters[3];
  bSplineKnotOrigin[1] = fixedParameters[4];
  bSplineKnotOrigin[2] = fixedParameters[5];

  typename FixedImageType::SpacingType bSplineKnotSpacing;
  bSplineKnotSpacing[0] = fixedParameters[6];
  bSplineKnotSpacing[1] = fixedParameters[7];
  bSplineKnotSpacing[2] = fixedParameters[8];

  typename FixedImageType::RegionType bSplineKnotRegion;
  bSplineKnotRegion.SetSize(bSplineKnotSize);

  this->m_BSplineKnotImage->SetRegions(bSplineKnotRegion);
  this->m_BSplineKnotImage->SetSpacing(bSplineKnotSpacing);
  this->m_BSplineKnotImage->SetOrigin(bSplineKnotOrigin);
  this->m_BSplineKnotImage->SetDirection(this->m_FixedImage->GetDirection());
  this->m_BSplineKnotImage->Update();

  /** Initialize PenaltyGridImage. */
  this->m_PenaltyGridImage = PenaltyGridImageType::New();

  typename SegmentedImageType::RegionType sampledSegmentedImageRegion =
    this->m_SampledSegmentedImage->GetBufferedRegion();
  typename SegmentedImageType::PointType     sampledSegmentedImageOrigin = this->m_SampledSegmentedImage->GetOrigin();
  typename SegmentedImageType::SpacingType   sampledSegmentedImageSpacing = this->m_SampledSegmentedImage->GetSpacing();
  typename SegmentedImageType::DirectionType sampledSegmentedImageDirection =
    this->m_SampledSegmentedImage->GetDirection();

  this->m_PenaltyGridImage->SetRegions(sampledSegmentedImageRegion);
  this->m_PenaltyGridImage->SetSpacing(sampledSegmentedImageSpacing);
  this->m_PenaltyGridImage->SetOrigin(sampledSegmentedImageOrigin);
  this->m_PenaltyGridImage->SetDirection(sampledSegmentedImageDirection);
  this->m_PenaltyGridImage->Update();

  /** compute number of knots in rigid regions */
  this->m_NumberOfRigidGrids = 0;

  using PenaltyGridIteratorType = itk::ImageRegionIterator<PenaltyGridImageType>;
  PenaltyGridIteratorType ki(this->m_PenaltyGridImage, this->m_PenaltyGridImage->GetBufferedRegion());

  typename PenaltyGridImageType::IndexType penaltyGridIndex;
  typename PenaltyGridImageType::PointType penaltyGridPoint;

  // typedef itk::LinearInterpolateImageFunction< SegmentedImageType, double > SegmentedImageInterpolatorType;
  using SegmentedImageInterpolatorType = itk::NearestNeighborInterpolateImageFunction<SegmentedImageType, double>;
  auto segmentedImageInterpolator = SegmentedImageInterpolatorType::New();

  segmentedImageInterpolator->SetInputImage(this->m_SampledSegmentedImage);

  unsigned int PixelValue;

  ki.GoToBegin();

  while (!ki.IsAtEnd())
  {
    penaltyGridIndex = ki.GetIndex();

    this->m_PenaltyGridImage->TransformIndexToPhysicalPoint(penaltyGridIndex, penaltyGridPoint);

    PixelValue = static_cast<unsigned int>(segmentedImageInterpolator->Evaluate(penaltyGridPoint));

    if (PixelValue > 0)
    {
      this->m_NumberOfRigidGrids++;
    }
    ++ki;
  }
} // end Initialize()


/**
 * *********************** GetValue *****************************
 */

template <class TFixedImage, class TScalarType>
auto
DistancePreservingRigidityPenaltyTerm<TFixedImage, TScalarType>::GetValue(const ParametersType & parameters) const
  -> MeasureType
{
  /** Set output values to zero. */
  this->m_RigidityPenaltyTermValue = NumericTraits<MeasureType>::Zero;

  // this->SetTransformParameters( parameters );
  this->m_BSplineTransform->SetParameters(parameters);

  /** distance-preserving penalty computation */
  MeasureType penaltyTerm = 0.0;
  MeasureType penaltyTermBuffer = 0.0;

  MeasureType dx = 0.0;
  MeasureType dX = 0.0;

  using PenaltyGridIteratorType = itk::ImageRegionConstIteratorWithIndex<PenaltyGridImageType>;
  PenaltyGridImageRegionType penaltyGridImageRegion = this->m_PenaltyGridImage->GetBufferedRegion();

  PenaltyGridIteratorType pgi(this->m_PenaltyGridImage, penaltyGridImageRegion);

  typename PenaltyGridImageType::IndexType penaltyGridIndex, neighborPenaltyGridIndex;
  typename PenaltyGridImageType::PointType penaltyGridPoint, neighborPenaltyGridPoint, xn, xf;

  using NeighborhoodIteratorType = itk::ConstNeighborhoodIterator<PenaltyGridImageType>;
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);
  NeighborhoodIteratorType ni(radius, this->m_PenaltyGridImage, penaltyGridImageRegion);
  unsigned int             numberOfNeighborhood = ni.Size();

  unsigned int numberOfRigidGridsNeighbor;

  // interpolation of segmented image
  using SegmentedImageInterpolatorType = itk::NearestNeighborInterpolateImageFunction<SegmentedImageType, double>;
  auto segmentedImageInterpolator = SegmentedImageInterpolatorType::New();

  segmentedImageInterpolator->SetInputImage(this->m_SampledSegmentedImage);

  unsigned int pixelValue, pixelValueNeighbor;

  /** Penalty term computation */
  if (MovingImageDimension == 3)
  {
    for (pgi.GoToBegin(), ni.GoToBegin(); !pgi.IsAtEnd(); ++pgi, ++ni)
    {
      penaltyGridIndex = pgi.GetIndex();

      this->m_PenaltyGridImage->TransformIndexToPhysicalPoint(penaltyGridIndex, penaltyGridPoint);

      ni.SetLocation(penaltyGridIndex);

      pixelValue = static_cast<unsigned int>(segmentedImageInterpolator->Evaluate(penaltyGridPoint));

      if (pixelValue > 0 && pixelValue < 6)
      {
        numberOfRigidGridsNeighbor = 0;

        for (unsigned int kk = 0; kk < numberOfNeighborhood; ++kk)
        {
          neighborPenaltyGridIndex = ni.GetIndex(kk);

          this->m_PenaltyGridImage->TransformIndexToPhysicalPoint(neighborPenaltyGridIndex, neighborPenaltyGridPoint);

          pixelValueNeighbor =
            static_cast<unsigned int>(segmentedImageInterpolator->Evaluate(neighborPenaltyGridPoint));

          if (pixelValue == pixelValueNeighbor)
          {
            ++numberOfRigidGridsNeighbor;
          }
        }

        if (numberOfRigidGridsNeighbor > 1)
        {
          for (unsigned int kk = 0; kk < numberOfNeighborhood; ++kk)
          {
            neighborPenaltyGridIndex = ni.GetIndex(kk);

            this->m_PenaltyGridImage->TransformIndexToPhysicalPoint(neighborPenaltyGridIndex, neighborPenaltyGridPoint);

            pixelValueNeighbor =
              static_cast<unsigned int>(segmentedImageInterpolator->Evaluate(neighborPenaltyGridPoint));

            if (pixelValue == pixelValueNeighbor)
            {
              xn = this->m_Transform->TransformPoint(neighborPenaltyGridPoint);
              xf = this->m_Transform->TransformPoint(penaltyGridPoint);

              dX = (neighborPenaltyGridPoint[0] - penaltyGridPoint[0]) *
                     (neighborPenaltyGridPoint[0] - penaltyGridPoint[0]) +
                   (neighborPenaltyGridPoint[1] - penaltyGridPoint[1]) *
                     (neighborPenaltyGridPoint[1] - penaltyGridPoint[1]) +
                   (neighborPenaltyGridPoint[2] - penaltyGridPoint[2]) *
                     (neighborPenaltyGridPoint[2] - penaltyGridPoint[2]);

              dx = (xn[0] - xf[0]) * (xn[0] - xf[0]) + (xn[1] - xf[1]) * (xn[1] - xf[1]) +
                   (xn[2] - xf[2]) * (xn[2] - xf[2]);

              penaltyTermBuffer = (dx - dX) * (dx - dX);

              penaltyTerm += penaltyTermBuffer / numberOfRigidGridsNeighbor / (this->m_NumberOfRigidGrids);
            }
          } // end for
        }   // end if
      }     // end if
    }       // end for
  }         // end if ( dimension = 3 )

  /** Return the rigidity penalty term value. */
  return penaltyTerm;

} // end GetValue()


/**
 * *********************** GetDerivative ************************
 */

template <class TFixedImage, class TScalarType>
void
DistancePreservingRigidityPenaltyTerm<TFixedImage, TScalarType>::GetDerivative(const ParametersType & parameters,
                                                                               DerivativeType &       derivative) const
{
  MeasureType dummyvalue = NumericTraits<MeasureType>::Zero;
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);
} // end GetDerivative()


/**
 * *********************** GetValueAndDerivative ****************
 */

template <class TFixedImage, class TScalarType>
void
DistancePreservingRigidityPenaltyTerm<TFixedImage, TScalarType>::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  /** Set output values to zero. */
  value = NumericTraits<MeasureType>::Zero;
  this->m_RigidityPenaltyTermValue = NumericTraits<MeasureType>::Zero;

  /** Set output values to zero. */
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits<MeasureType>::ZeroValue());

  this->m_BSplineTransform->SetParameters(parameters);

  /** Distance-preserving penalty */
  MeasureType penaltyTermBuffer = 0.0;
  MeasureType derivativeTermTemp1 = 0.0, derivativeTermTemp2 = 0.0, derivativeTermTemp3 = 0.0;

  unsigned int par1, par2;

  MeasureType dX = 0.0;
  MeasureType dx = 0.0;

  typename PenaltyGridImageType::IndexType penaltyGridIndex, neighborPenaltyGridIndex;
  typename PenaltyGridImageType::PointType penaltyGridPoint, neighborPenaltyGridPoint, xn, xf;

  using BSplineKernelFunctionType = itk::BSplineKernelFunction<3>;
  using WeightsFunctionType = itk::BSplineInterpolationWeightFunction<double, ImageDimension, 3>;
  using ContinuousIndexType = typename WeightsFunctionType::ContinuousIndexType;
  using ContinuousIndexValueType = double;

  ContinuousIndexValueType tx, ty, tz;

  ContinuousIndexValueType m, n, p, neighbor_m, neighbor_n, neighbor_p;

  using PenaltyGridIteratorType = itk::ImageRegionConstIteratorWithIndex<PenaltyGridImageType>;
  PenaltyGridImageRegionType penaltyGridImageRegion = this->m_PenaltyGridImage->GetBufferedRegion();
  PenaltyGridIteratorType    pgi(this->m_PenaltyGridImage, penaltyGridImageRegion);

  // neighborhood iterator
  using NeighborhoodIteratorType = itk::ConstNeighborhoodIterator<PenaltyGridImageType>;
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);
  NeighborhoodIteratorType ni(radius, this->m_PenaltyGridImage, penaltyGridImageRegion);
  unsigned int             numberOfNeighborhood = ni.Size();

  ContinuousIndexType      ntindex_start, ntindex_neighbor_start;
  ContinuousIndexValueType tx_neighbor, ty_neighbor, tz_neighbor;

  typename BSplineKnotImageType::SizeType bSplineKnotImageSize =
    this->m_BSplineKnotImage->GetBufferedRegion().GetSize();

  MeasureType du_dC, du_dC_neighbor;

  unsigned int numberOfRigidGridsNeighbor;

  const unsigned int parametersDimension = this->GetNumberOfParameters();
  unsigned int       numberOfParametersPerDimension = parametersDimension / ImageDimension;

  // interpolation of segmented image
  using SegmentedImageInterpolatorType = itk::NearestNeighborInterpolateImageFunction<SegmentedImageType, double>;
  auto segmentedImageInterpolator = SegmentedImageInterpolatorType::New();

  segmentedImageInterpolator->SetInputImage(this->m_SampledSegmentedImage);

  unsigned int pixelValue, pixelValueNeighbor;

  if (MovingImageDimension == 3)
  {
    for (pgi.GoToBegin(), ni.GoToBegin(); !pgi.IsAtEnd(); ++pgi, ++ni)
    {
      penaltyGridIndex = pgi.GetIndex();

      this->m_PenaltyGridImage->TransformIndexToPhysicalPoint(penaltyGridIndex, penaltyGridPoint);

      pixelValue = static_cast<unsigned int>(segmentedImageInterpolator->Evaluate(penaltyGridPoint));

      ni.SetLocation(penaltyGridIndex);

      /** Penalty term calculation */
      if (pixelValue > 0 && pixelValue < 6)
      {
        numberOfRigidGridsNeighbor = 0;

        for (unsigned int kk = 0; kk < numberOfNeighborhood; ++kk)
        {
          neighborPenaltyGridIndex = ni.GetIndex(kk);

          this->m_PenaltyGridImage->TransformIndexToPhysicalPoint(neighborPenaltyGridIndex, neighborPenaltyGridPoint);

          pixelValueNeighbor =
            static_cast<unsigned int>(segmentedImageInterpolator->Evaluate(neighborPenaltyGridPoint));

          if (pixelValue == pixelValueNeighbor)
          {
            ++numberOfRigidGridsNeighbor;
          }
        }

        if (numberOfRigidGridsNeighbor > 1)
        {
          for (unsigned int kk = 0; kk < numberOfNeighborhood; ++kk)
          {
            neighborPenaltyGridIndex = ni.GetIndex(kk);

            this->m_PenaltyGridImage->TransformIndexToPhysicalPoint(neighborPenaltyGridIndex, neighborPenaltyGridPoint);

            pixelValueNeighbor =
              static_cast<unsigned int>(segmentedImageInterpolator->Evaluate(neighborPenaltyGridPoint));

            if (pixelValue == pixelValueNeighbor)
            {
              xn = this->m_Transform->TransformPoint(neighborPenaltyGridPoint);
              xf = this->m_Transform->TransformPoint(penaltyGridPoint);

              dX = (neighborPenaltyGridPoint[0] - penaltyGridPoint[0]) *
                     (neighborPenaltyGridPoint[0] - penaltyGridPoint[0]) +
                   (neighborPenaltyGridPoint[1] - penaltyGridPoint[1]) *
                     (neighborPenaltyGridPoint[1] - penaltyGridPoint[1]) +
                   (neighborPenaltyGridPoint[2] - penaltyGridPoint[2]) *
                     (neighborPenaltyGridPoint[2] - penaltyGridPoint[2]);

              dx = (xn[0] - xf[0]) * (xn[0] - xf[0]) + (xn[1] - xf[1]) * (xn[1] - xf[1]) +
                   (xn[2] - xf[2]) * (xn[2] - xf[2]);

              penaltyTermBuffer = (dx - dX) * (dx - dX);

              value += penaltyTermBuffer / numberOfRigidGridsNeighbor / (this->m_NumberOfRigidGrids);

              // find neighboring B-Spline control points
              const auto tindex =
                this->m_BSplineKnotImage->template TransformPhysicalPointToContinuousIndex<ContinuousIndexValueType>(
                  penaltyGridPoint);
              const auto tindex_neighbor =
                this->m_BSplineKnotImage->template TransformPhysicalPointToContinuousIndex<ContinuousIndexValueType>(
                  neighborPenaltyGridPoint);

              tx = tindex[0];
              ty = tindex[1];
              tz = tindex[2];

              tx_neighbor = tindex_neighbor[0];
              ty_neighbor = tindex_neighbor[1];
              tz_neighbor = tindex_neighbor[2];

              for (unsigned dd = 0; dd < ImageDimension; ++dd)
              {
                ntindex_start[dd] = static_cast<ContinuousIndexValueType>(floor(tindex[dd])) - 1.0;
                ntindex_neighbor_start[dd] = static_cast<ContinuousIndexValueType>(floor(tindex_neighbor[dd])) - 1.0;
              }

              derivativeTermTemp1 =
                4 * (dx - dX) * (xn[0] - xf[0]) / numberOfRigidGridsNeighbor / (this->m_NumberOfRigidGrids);
              derivativeTermTemp2 =
                4 * (dx - dX) * (xn[1] - xf[1]) / numberOfRigidGridsNeighbor / (this->m_NumberOfRigidGrids);
              derivativeTermTemp3 =
                4 * (dx - dX) * (xn[2] - xf[2]) / numberOfRigidGridsNeighbor / (this->m_NumberOfRigidGrids);

              for (unsigned int kk = 0; kk < 4; ++kk)
              {
                p = ntindex_start[2] + kk;
                neighbor_p = ntindex_neighbor_start[2] + kk;

                for (unsigned int jj = 0; jj < 4; ++jj)
                {
                  n = ntindex_start[1] + jj;
                  neighbor_n = ntindex_neighbor_start[1] + jj;

                  for (unsigned int ii = 0; ii < 4; ++ii)
                  {
                    m = ntindex_start[0] + ii;
                    neighbor_m = ntindex_neighbor_start[0] + ii;

                    // neighborhood of (i',j',k')
                    du_dC_neighbor = (BSplineKernelFunctionType::FastEvaluate(tx_neighbor - neighbor_m)) *
                                     (BSplineKernelFunctionType::FastEvaluate(ty_neighbor - neighbor_n)) *
                                     (BSplineKernelFunctionType::FastEvaluate(tz_neighbor - neighbor_p));

                    par1 = static_cast<unsigned int>(neighbor_m) +
                           bSplineKnotImageSize[0] * static_cast<unsigned int>(neighbor_n) +
                           bSplineKnotImageSize[0] * bSplineKnotImageSize[1] * static_cast<unsigned int>(neighbor_p);

                    derivative[par1] += derivativeTermTemp1 * du_dC_neighbor;
                    derivative[par1 + numberOfParametersPerDimension] += derivativeTermTemp2 * du_dC_neighbor;
                    derivative[par1 + 2 * numberOfParametersPerDimension] += derivativeTermTemp3 * du_dC_neighbor;

                    // neighborhood of (i,j,k)
                    du_dC = (BSplineKernelFunctionType::FastEvaluate(tx - m)) *
                            (BSplineKernelFunctionType::FastEvaluate(ty - n)) *
                            (BSplineKernelFunctionType::FastEvaluate(tz - p));

                    par2 = static_cast<unsigned int>(m) + bSplineKnotImageSize[0] * static_cast<unsigned int>(n) +
                           bSplineKnotImageSize[0] * bSplineKnotImageSize[1] * static_cast<unsigned int>(p);

                    derivative[par2] -= derivativeTermTemp1 * du_dC;
                    derivative[par2 + numberOfParametersPerDimension] -= derivativeTermTemp2 * du_dC;
                    derivative[par2 + 2 * numberOfParametersPerDimension] -= derivativeTermTemp3 * du_dC;
                  }
                }
              }
            }
          } // end for
        }   // end if
      }
    } // end for
  }   // end if ( dimension = 3 )

} // end GetValueAndDerivative()


/**
 * ********************* PrintSelf ******************************
 */

template <class TFixedImage, class TScalarType>
void
DistancePreservingRigidityPenaltyTerm<TFixedImage, TScalarType>::PrintSelf(std::ostream & os, Indent indent) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf(os, indent);

  /** Add debugging information. */
  os << indent << "BSplineTransform: " << this->m_BSplineTransform << std::endl;
  os << indent << "RigidityPenaltyTermValue: " << this->m_RigidityPenaltyTermValue << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // #ifndef itkDistancePreservingRigidityPenaltyTerm_hxx
