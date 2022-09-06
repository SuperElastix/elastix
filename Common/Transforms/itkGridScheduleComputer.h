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
#ifndef itkGridScheduleComputer_h
#define itkGridScheduleComputer_h

#include "itkObject.h"
#include "itkImageBase.h"
#include "itkTransform.h"

namespace itk
{

/**
 * \class GridScheduleComputer
 * \brief This class computes all information about the B-spline grid,
 * given the image information and the desired grid spacing.
 *
 * NB: the Direction Cosines of the B-spline grid are set identical
 * to the user-supplied ImageDirection.
 *
 * \ingroup Transforms
 */

template <typename TTransformScalarType, unsigned int VImageDimension>
class ITK_TEMPLATE_EXPORT GridScheduleComputer : public Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GridScheduleComputer);

  /** Standard class typedefs. */
  using Self = GridScheduleComputer;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GridScheduleComputer, Object);

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
  using VectorDirectionType = std::vector<DirectionType>;
  using VectorRegionType = std::vector<RegionType>;
  using VectorGridSpacingFactorType = std::vector<GridSpacingFactorType>;

  /** Typedefs for the initial transform. */
  using TransformType = Transform<TransformScalarType, Self::Dimension, Self::Dimension>;
  using TransformPointer = typename TransformType::Pointer;
  using TransformConstPointer = typename TransformType::ConstPointer;

  /** Set the ImageOrigin. */
  itkSetMacro(ImageOrigin, OriginType);

  /** Get the ImageOrigin. */
  itkGetConstMacro(ImageOrigin, OriginType);

  /** Set the ImageSpacing. */
  itkSetMacro(ImageSpacing, SpacingType);

  /** Get the ImageSpacing. */
  itkGetConstMacro(ImageSpacing, SpacingType);

  /** Set the ImageDirection. */
  itkSetMacro(ImageDirection, DirectionType);

  /** Get the ImageDirection. */
  itkGetConstMacro(ImageDirection, DirectionType);

  /** Set the ImageRegion. */
  itkSetMacro(ImageRegion, RegionType);

  /** Get the ImageRegion. */
  itkGetConstMacro(ImageRegion, RegionType);

  /** Set the B-spline order. */
  itkSetClampMacro(BSplineOrder, unsigned int, 0, 5);

  /** Get the B-spline order. */
  itkGetConstMacro(BSplineOrder, unsigned int);

  /** Set the final grid spacing. */
  itkSetMacro(FinalGridSpacing, SpacingType);

  /** Get the final grid spacing. */
  itkGetConstMacro(FinalGridSpacing, SpacingType);

  /** Set a default grid spacing schedule. */
  virtual void
  SetDefaultSchedule(unsigned int levels, double upsamplingFactor);

  /** Set a grid spacing schedule. */
  virtual void
  SetSchedule(const VectorGridSpacingFactorType & schedule);

  /** Get the grid spacing schedule. */
  virtual void
  GetSchedule(VectorGridSpacingFactorType & schedule) const;

  /** Set an initial Transform. Only set one if composition is used. */
  itkSetConstObjectMacro(InitialTransform, TransformType);

  /** Compute the B-spline grid. */
  virtual void
  ComputeBSplineGrid();

  /** Get the B-spline grid at some level. */
  virtual void
  GetBSplineGrid(unsigned int    level,
                 RegionType &    gridRegion,
                 SpacingType &   gridSpacing,
                 OriginType &    gridOrigin,
                 DirectionType & gridDirection);

protected:
  /** The constructor. */
  GridScheduleComputer();

  /** The destructor. */
  ~GridScheduleComputer() override = default;

  /** Declare member variables, needed for B-spline grid. */
  VectorSpacingType           m_GridSpacings;
  VectorOriginType            m_GridOrigins;
  VectorDirectionType         m_GridDirections;
  VectorRegionType            m_GridRegions;
  TransformConstPointer       m_InitialTransform;
  VectorGridSpacingFactorType m_GridSpacingFactors;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Get number of levels. */
  itkGetConstMacro(NumberOfLevels, unsigned int);

  /** Function to apply the initial transform, if it exists. */
  virtual void
  ApplyInitialTransform(OriginType &    imageOrigin,
                        SpacingType &   imageSpacing,
                        DirectionType & imageDirection,
                        SpacingType &   finalGridSpacing) const;

private:
  /** Declare member variables, needed in functions. */
  OriginType    m_ImageOrigin;
  SpacingType   m_ImageSpacing;
  RegionType    m_ImageRegion;
  DirectionType m_ImageDirection;
  unsigned int  m_BSplineOrder;
  unsigned int  m_NumberOfLevels;
  SpacingType   m_FinalGridSpacing;

  /** Clamp the upsampling factor. */
  itkSetClampMacro(UpsamplingFactor, float, 1.0, NumericTraits<float>::max());

  /** Declare member variables, needed internally. */
  float m_UpsamplingFactor;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGridScheduleComputer.hxx"
#endif

#endif // end #ifndef itkGridScheduleComputer_h
