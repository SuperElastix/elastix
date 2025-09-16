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
#ifndef itkGridScheduleComputer_hxx
#define itkGridScheduleComputer_hxx

#include "itkGridScheduleComputer.h"

#include "itkImageRegionExclusionConstIteratorWithIndex.h"
#include "itkConfigure.h"
#include <vnl/vnl_inverse.h>
#include "itkBoundingBox.h"
#include <cmath> // For pow.

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template <typename TTransformScalarType, unsigned int VImageDimension>
GridScheduleComputer<TTransformScalarType, VImageDimension>::GridScheduleComputer()
{
  this->SetDefaultSchedule(3);

} // end Constructor()


/**
 * ********************* SetDefaultSchedule ****************************
 */

template <typename TTransformScalarType, unsigned int VImageDimension>
void
GridScheduleComputer<TTransformScalarType, VImageDimension>::SetDefaultSchedule(unsigned int numberOfLevels)
{
  // Set member variable.
  m_NumberOfLevels = numberOfLevels;

  // Initialize the schedule.
  m_GridSpacingFactors.resize(numberOfLevels);

  // Set up a default schedule, having consecutive decreasing powers of two as sampling factors.
  for (unsigned int level{}; level < numberOfLevels; ++level)
  {
    m_GridSpacingFactors[level] = MakeFilled<GridSpacingFactorType>(std::pow(2.0, numberOfLevels - level - 1));
  }

} // end SetDefaultSchedule()


/**
 * ********************* SetSchedule ****************************
 */

template <typename TTransformScalarType, unsigned int VImageDimension>
void
GridScheduleComputer<TTransformScalarType, VImageDimension>::SetSchedule(const VectorGridSpacingFactorType & schedule)
{
  /** Set member variables. */
  m_GridSpacingFactors = schedule;
  m_NumberOfLevels = schedule.size();

} // end SetSchedule()


/**
 * ********************* GetSchedule ****************************
 */

template <typename TTransformScalarType, unsigned int VImageDimension>
auto
GridScheduleComputer<TTransformScalarType, VImageDimension>::GetSchedule() const -> VectorGridSpacingFactorType
{
  return m_GridSpacingFactors;

} // end GetSchedule()


/**
 * ********************* ComputeBSplineGrid ****************************
 */

template <typename TTransformScalarType, unsigned int VImageDimension>
void
GridScheduleComputer<TTransformScalarType, VImageDimension>::ComputeBSplineGrid()
{
  OriginType    imageOrigin;
  SpacingType   imageSpacing, finalGridSpacing;
  DirectionType imageDirection;

  /** Apply the initial transform.  */
  this->ApplyInitialTransform(imageOrigin, imageSpacing, imageDirection, finalGridSpacing);

  /** Set the appropriate sizes. */
  m_GridOrigins.resize(m_NumberOfLevels);
  m_GridRegions.resize(m_NumberOfLevels);
  m_GridSpacings.resize(m_NumberOfLevels);
  m_GridDirections.resize(m_NumberOfLevels);

  /** For all levels ... */
  for (unsigned int res = 0; res < m_NumberOfLevels; ++res)
  {
    /** For all dimensions ... */
    SizeType size = m_ImageRegion.GetSize();
    SizeType gridsize;
    for (unsigned int dim = 0; dim < Dimension; ++dim)
    {
      /** Compute the grid spacings. */
      double gridSpacing = finalGridSpacing[dim] * m_GridSpacingFactors[res][dim];
      m_GridSpacings[res][dim] = gridSpacing;

      /** Compute the grid size without the extra grid points at the edges. */
      const auto bareGridSize = static_cast<unsigned int>(std::ceil(size[dim] * imageSpacing[dim] / gridSpacing));

      /** The number of B-spline grid nodes is the bareGridSize plus the
       * B-spline order more grid nodes. */
      gridsize[dim] = static_cast<SizeValueType>(bareGridSize + m_BSplineOrder);

      /** Compute the origin of the B-spline grid. */
      m_GridOrigins[res][dim] =
        imageOrigin[dim] - ((gridsize[dim] - 1) * gridSpacing - (size[dim] - 1) * imageSpacing[dim]) / 2.0;
    }

    /** Take into account direction cosines:
     * rotate grid origin around image origin. */
    m_GridOrigins[res] = imageOrigin + imageDirection * (m_GridOrigins[res] - imageOrigin);

    /** Set the grid region. */
    m_GridRegions[res].SetSize(gridsize);

    /** Simply copy the image direction for now */
    m_GridDirections[res] = imageDirection;
  }

} // end ComputeBSplineGrid()


/**
 * ********************* ApplyInitialTransform ****************************
 *
 * This function adapts the m_ImageOrigin and m_ImageSpacing.
 * This makes sure that the B-spline grid is located at the position
 * of the fixed image after undergoing the initial transform.
 */

template <typename TTransformScalarType, unsigned int VImageDimension>
void
GridScheduleComputer<TTransformScalarType, VImageDimension>::ApplyInitialTransform(OriginType &    imageOrigin,
                                                                                   SpacingType &   imageSpacing,
                                                                                   DirectionType & imageDirection,
                                                                                   SpacingType & finalGridSpacing) const
{
  /** Check for the existence of an initial transform. */
  if (m_InitialTransform.IsNull())
  {
    imageOrigin = m_ImageOrigin;
    imageSpacing = m_ImageSpacing;
    imageDirection = m_ImageDirection;
    finalGridSpacing = m_FinalGridSpacing;
    return;
  }

  /** We could rotate the image direction according
   * to the average rotation of the initial transformation.
   * For now leave it as is. */
  imageDirection = m_ImageDirection;
  typename DirectionType::InternalMatrixType invImageDirectionTemp = vnl_inverse(imageDirection.GetVnlMatrix());
  DirectionType                              invImageDirection(invImageDirectionTemp);

  /** We have to determine a bounding box around the fixed image after
   * applying the initial transform. This is done by iterating over the
   * the boundary of the fixed image, evaluating the initial transform
   * at those points, and keeping track of the minimum/maximum transformed
   * coordinate in each dimension.
   *
   * NB: the possibility of non-identity direction cosines makes it
   * a bit more complicated. This is dealt with by applying the inverse
   * direction matrix during computation of the bounding box.
   *
   * \todo: automatically estimate an optimal imageDirection?
   */

  /** Create a temporary image. As small as possible, for memory savings. */
  using ImageType = Image<unsigned char, Dimension>; // bool??
  auto image = ImageType::New();
  image->SetOrigin(m_ImageOrigin);
  image->SetSpacing(m_ImageSpacing);
  image->SetDirection(m_ImageDirection);
  image->SetRegions(m_ImageRegion);
  image->Allocate();

  /** The points that define the bounding box. */
  using PointValueType = typename PointType::ValueType;
  using BoundingBoxType = BoundingBox<unsigned long, Dimension, PointValueType>;
  using BoundingBoxPointer = typename BoundingBoxType::Pointer;
  using PointsContainerType = typename BoundingBoxType::PointsContainer;
  using PointsContainerPointer = typename PointsContainerType::Pointer;
  BoundingBoxPointer     boundingBox = BoundingBoxType::New();
  PointsContainerPointer boundaryPoints = PointsContainerType::New();
  OriginType             maxPoint;
  OriginType             minPoint;
  maxPoint.Fill(NumericTraits<TransformScalarType>::NonpositiveMin());
  minPoint.Fill(NumericTraits<TransformScalarType>::max());

  /** An iterator over the boundary of the image. */
  using BoundaryIteratorType = ImageRegionExclusionConstIteratorWithIndex<ImageType>;
  BoundaryIteratorType bit(image, m_ImageRegion);
  bit.SetExclusionRegionToInsetRegion();
  bit.GoToBegin();
  SizeType imageSize = m_ImageRegion.GetSize();
  SizeType insetImageSize = imageSize;
  for (unsigned int i = 0; i < Dimension; ++i)
  {
    if (insetImageSize[i] > 1)
    {
      insetImageSize[i] -= 2;
    }
    else
    {
      insetImageSize[i] = 0;
    }
  }
  RegionType          insetImageRegion(insetImageSize);
  const unsigned long numberOfBoundaryPoints = m_ImageRegion.GetNumberOfPixels() - insetImageRegion.GetNumberOfPixels();
  boundaryPoints->reserve(numberOfBoundaryPoints);

  /** Start loop over boundary and compute transformed points. */
  using IndexType = typename ImageType::IndexType;
  while (!bit.IsAtEnd())
  {
    /** Get index, transform to physical point, apply initial transform.
     * NB: the OutputPointType of the initial transform by definition equals
     * the InputPointType of this transform.
     */
    IndexType  inputIndex = bit.GetIndex();
    OriginType inputPoint;
    image->TransformIndexToPhysicalPoint(inputIndex, inputPoint);
    typename TransformType::OutputPointType outputPoint = m_InitialTransform->TransformPoint(inputPoint);

    // CHECK: shouldn't TransformIndexToPhysicalPoint do this?
    outputPoint = invImageDirection * outputPoint;

    /** Store transformed point */
    boundaryPoints->push_back(outputPoint);

    /** Step to next voxel. */
    ++bit;

  } // end while loop over image boundary

  /** Compute min and max point */
  boundingBox->SetPoints(boundaryPoints);
  boundingBox->ComputeBoundingBox();
  minPoint = boundingBox->GetMinimum();
  maxPoint = boundingBox->GetMaximum();

  /** Set minPoint as the new "ImageOrigin" (between quotes, since it
   * is not really the origin of the fixedImage anymore).
   * Take into account direction cosines */
  imageOrigin = minPoint;
  imageOrigin = imageDirection * imageOrigin;

  /** Compute the new "ImageSpacing" in each dimension. */
  const double smallnumber = NumericTraits<double>::epsilon();
  for (unsigned int i = 0; i < Dimension; ++i)
  {
    /** Compute the length of the fixed image (in mm) for dimension i. */
    double oldLength_i = m_ImageSpacing[i] * static_cast<double>(m_ImageRegion.GetSize()[i] - 1);

    /** Compute the length of the bounding box (in mm) for dimension i. */
    auto newLength_i = static_cast<double>(maxPoint[i] - minPoint[i]);

    /** Scale the fixedImageSpacing by their ratio. */
    if (oldLength_i > smallnumber)
    {
      imageSpacing[i] = m_ImageSpacing[i] * (newLength_i / oldLength_i);
      finalGridSpacing[i] = m_FinalGridSpacing[i] * (newLength_i / oldLength_i);
    }
  }

} // end ApplyInitialTransform()


/**
 * ********************* GetBSplineGrid ****************************
 */

template <typename TTransformScalarType, unsigned int VImageDimension>
void
GridScheduleComputer<TTransformScalarType, VImageDimension>::GetBSplineGrid(unsigned int    level,
                                                                            RegionType &    gridRegion,
                                                                            SpacingType &   gridSpacing,
                                                                            OriginType &    gridOrigin,
                                                                            DirectionType & gridDirection)
{
  /** Check level. */
  if (level > m_NumberOfLevels - 1)
  {
    itkExceptionMacro("ERROR: Requesting resolution level " << level << ", but only " << m_NumberOfLevels
                                                            << " levels exist.");
  }

  /** Return values. */
  gridRegion = m_GridRegions[level];
  gridSpacing = m_GridSpacings[level];
  gridOrigin = m_GridOrigins[level];
  gridDirection = m_GridDirections[level];

} // end GetBSplineGrid()


/**
 * ********************* PrintSelf ****************************
 */

template <typename TTransformScalarType, unsigned int VImageDimension>
void
GridScheduleComputer<TTransformScalarType, VImageDimension>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "B-spline order: " << m_BSplineOrder << std::endl;
  os << indent << "NumberOfLevels: " << m_NumberOfLevels << std::endl;

  os << indent << "ImageSpacing: " << m_ImageSpacing << std::endl;
  os << indent << "ImageOrigin: " << m_ImageOrigin << std::endl;
  os << indent << "ImageDirection: " << m_ImageDirection << std::endl;
  os << indent << "ImageRegion: " << std::endl;
  m_ImageRegion.Print(os, indent.GetNextIndent());

  os << indent << "FinalGridSpacing: " << m_FinalGridSpacing << std::endl;
  os << indent << "GridSpacingFactors: " << std::endl;
  for (unsigned int i = 0; i < m_NumberOfLevels; ++i)
  {
    os << indent.GetNextIndent() << m_GridSpacingFactors[i] << std::endl;
  }

  os << indent << "GridSpacings: " << std::endl;
  for (unsigned int i = 0; i < m_NumberOfLevels; ++i)
  {
    os << indent.GetNextIndent() << m_GridSpacings[i] << std::endl;
  }

  os << indent << "GridOrigins: " << std::endl;
  for (unsigned int i = 0; i < m_NumberOfLevels; ++i)
  {
    os << indent.GetNextIndent() << m_GridOrigins[i] << std::endl;
  }

  os << indent << "GridDirections: " << std::endl;
  for (unsigned int i = 0; i < m_NumberOfLevels; ++i)
  {
    os << indent.GetNextIndent() << m_GridDirections[i] << std::endl;
  }

  os << indent << "GridRegions: " << std::endl;
  for (unsigned int i = 0; i < m_NumberOfLevels; ++i)
  {
    os << indent.GetNextIndent() << m_GridRegions[i] << std::endl;
  }

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkGridScheduleComputer_hxx
