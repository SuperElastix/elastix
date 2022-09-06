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
#ifndef itkCyclicGridScheduleComputer_h
#define itkCyclicGridScheduleComputer_h

#include "itkImageBase.h"
#include "itkTransform.h"
#include "itkGridScheduleComputer.h"

namespace itk
{

/**
 * \class CyclicGridScheduleComputer
 *
 * \brief This class computes all information about the B-spline grid.
 *
 * This class computes all information about the B-spline grid
 * given the image information and the desired grid spacing. It differs from
 * the GridScheduleComputer in how the nodes are placed in the last dimension.
 *
 * \ingroup Transforms
 */

template <typename TTransformScalarType, unsigned int VImageDimension>
class ITK_TEMPLATE_EXPORT CyclicGridScheduleComputer
  : public GridScheduleComputer<TTransformScalarType, VImageDimension>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CyclicGridScheduleComputer);

  /** Standard class typedefs. */
  using Self = CyclicGridScheduleComputer;
  using Superclass = GridScheduleComputer<TTransformScalarType, VImageDimension>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CyclicGridScheduleComputer, GridScheduleComputer);

  /** Dimension of the domain space. */
  itkStaticConstMacro(Dimension, unsigned int, VImageDimension);

  /** Typedef's. */
  using TransformScalarType = TTransformScalarType;
  using ImageBaseType = ImageBase<Self::Dimension>;
  using PointType = typename ImageBaseType::PointType;
  using OriginType = typename ImageBaseType::PointType;
  using SpacingType = typename ImageBaseType::SpacingType;
  using DirectionType = typename ImageBaseType::DirectionType;
  using SizeType = typename ImageBaseType::SizeType;
  using SizeValueType = typename ImageBaseType::SizeValueType;
  using RegionType = typename ImageBaseType::RegionType;
  using GridSpacingFactorType = SpacingType;
  using VectorOriginType = std::vector<OriginType>;
  using VectorSpacingType = std::vector<SpacingType>;
  using VectorRegionType = std::vector<RegionType>;
  using VectorGridSpacingFactorType = std::vector<GridSpacingFactorType>;

  /** Typedefs for the initial transform. */
  using TransformType = Transform<TransformScalarType, Self::Dimension, Self::Dimension>;
  using TransformPointer = typename TransformType::Pointer;
  using TransformConstPointer = typename TransformType::ConstPointer;

  /** Compute the B-spline grid. */
  void
  ComputeBSplineGrid() override;

protected:
  /** The constructor. */
  CyclicGridScheduleComputer();

  /** The destructor. */
  ~CyclicGridScheduleComputer() override = default;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCyclicGridScheduleComputer.hxx"
#endif

#endif // end #ifndef itkCyclicGridScheduleComputer_h
