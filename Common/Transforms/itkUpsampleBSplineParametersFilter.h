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
#ifndef itkUpsampleBSplineParametersFilter_h
#define itkUpsampleBSplineParametersFilter_h

#include "itkObject.h"
#include "itkArray.h"

namespace itk
{

/** \class UpsampleBSplineParametersFilter
 *
 * \brief Convenience class for upsampling a B-spline coefficient image.
 *
 * The UpsampleBSplineParametersFilter class is a class that takes as input
 * the B-spline parameters. It's purpose is to compute new B-spline parameters
 * on a denser grid. Therefore, the user needs to supply the old B-spline grid
 * (region, spacing, origin, direction), and the required B-spline grid.
 *
 */

template <class TArray, class TImage>
class ITK_TEMPLATE_EXPORT UpsampleBSplineParametersFilter : public Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(UpsampleBSplineParametersFilter);

  /** Standard class typedefs. */
  using Self = UpsampleBSplineParametersFilter;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(UpsampleBSplineParametersFilter, Object);

  /** Typedefs. */
  using ArrayType = TArray;
  using ValueType = typename ArrayType::ValueType;
  using ImageType = TImage;
  using ImagePointer = typename ImageType::Pointer;
  using PixelType = typename ImageType::PixelType;
  using SpacingType = typename ImageType::SpacingType;
  using OriginType = typename ImageType::PointType;
  using DirectionType = typename ImageType::DirectionType;
  using RegionType = typename ImageType::RegionType;

  /** Dimension of the fixed image. */
  itkStaticConstMacro(Dimension, unsigned int, ImageType::ImageDimension);

  /** Set the origin of the current grid. */
  itkSetMacro(CurrentGridOrigin, OriginType);

  /** Set the spacing of the current grid. */
  itkSetMacro(CurrentGridSpacing, SpacingType);

  /** Set the direction of the current grid. */
  itkSetMacro(CurrentGridDirection, DirectionType);

  /** Set the region of the current grid. */
  itkSetMacro(CurrentGridRegion, RegionType);

  /** Set the origin of the required grid. */
  itkSetMacro(RequiredGridOrigin, OriginType);

  /** Set the spacing of the required grid. */
  itkSetMacro(RequiredGridSpacing, SpacingType);

  /** Set the direction of the required grid. */
  itkSetMacro(RequiredGridDirection, DirectionType);

  /** Set the region of the required grid. */
  itkSetMacro(RequiredGridRegion, RegionType);

  /** Set the B-spline order. */
  itkSetMacro(BSplineOrder, unsigned int);

  /** Compute the output parameter array. */
  virtual void
  UpsampleParameters(const ArrayType & param_in, ArrayType & param_out);

protected:
  /** Constructor. */
  UpsampleBSplineParametersFilter();

  /** Destructor. */
  ~UpsampleBSplineParametersFilter() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Function that checks if upsampling is required. */
  virtual bool
  DoUpsampling();

private:
  /** Private member variables. */
  OriginType    m_CurrentGridOrigin;
  SpacingType   m_CurrentGridSpacing;
  DirectionType m_CurrentGridDirection;
  RegionType    m_CurrentGridRegion;
  OriginType    m_RequiredGridOrigin;
  SpacingType   m_RequiredGridSpacing;
  DirectionType m_RequiredGridDirection;
  RegionType    m_RequiredGridRegion;
  unsigned int  m_BSplineOrder;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkUpsampleBSplineParametersFilter.hxx"
#endif

#endif // end #ifndef itkUpsampleBSplineParametersFilter_h
