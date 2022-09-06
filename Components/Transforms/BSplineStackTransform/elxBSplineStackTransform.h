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
#ifndef elxBSplineStackTransform_h
#define elxBSplineStackTransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning

/** Include itk transforms needed. */
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkBSplineStackTransform.h"

/** Include grid schedule computer and upsample filter. */
#include "itkGridScheduleComputer.h"
#include "itkUpsampleBSplineParametersFilter.h"

namespace elastix
{

/**
 * \class BSplineStackTransform
 * \brief A B-spline transform based on the itkStackTransform.
 *
 * This transform is a B-spline transformation, with for every time point a separate
 * D-1 dimensional B-spline transform. Calls to TransformPoint and GetJacobian are
 * redirected to the appropriate sub transform based on the last dimension (time) index.
 *
 * This transform uses the size, spacing and origin of the last dimension of the fixed
 * image to set the number of sub transforms the origin of the first transform and the
 * spacing between the transforms.
 *
 * When supplying the B-spline parameters (grid) make sure to provide dimension - 1 elements.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "BSplineStackTransform")</tt>
 * \parameter BSplineTransformSplineOrder: choose a B-spline order 1,2, or 3. \n
 *    example: <tt>(BSplineTransformSplineOrder 3)</tt>\n
 *    Default value: 3 (cubic B-splines).
 * \parameter FinalGridSpacingInVoxels: the grid spacing of the B-spline transform for each dimension. \n
 *    example: <tt>(FinalGridSpacingInVoxels 8.0 8.0 8.0)</tt> \n
 *    If only one argument is given, that factor is used for each dimension. The spacing
 *    is not in millimeters, but in "voxel size units".
 *    The default is 16.0 in every dimension.
 * \parameter FinalGridSpacingInPhysicalUnits: the grid spacing of the B-spline transform for each dimension. \n
 *    example: <tt>(FinalGridSpacingInPhysicalUnits 8.0 8.0 8.0)</tt> \n
 *    If only one argument is given, that factor is used for each dimension. The spacing
 *    is specified in millimeters.
 *    If not specified, the FinalGridSpacingInVoxels is used, or the FinalGridSpacing,
 *    to compute a FinalGridSpacingInPhysicalUnits. If those are not specified, the default
 *    value for FinalGridSpacingInVoxels is used to compute a FinalGridSpacingInPhysicalUnits.
 *    If an affine transformation is provided as initial transformation, the control grid
 *    will be scaled to cover the fixed image domain in the space defined by the initial transformation.
 * \parameter GridSpacingSchedule: the grid spacing downsampling factors for the B-spline transform
 *    for each dimension and each resolution. \n
 *    example: <tt>(GridSpacingSchedule 4.0 4.0 2.0 2.0 1.0 1.0)</tt> \n
 *    Which is an example for a 2D image, using 3 resolutions. \n
 *    For convenience, you may also specify only one value for each resolution:\n
 *    example: <tt>(GridSpacingSchedule 4.0 2.0 1.0 )</tt> \n
 *    which is equivalent to the example above.
 * \parameter PassiveEdgeWidth: the width of a band of control points at the border of the
 *   B-spline coefficient image that should remain passive during optimisation. \n
 *   Can be specified for each resolution. \n
 *   example: <tt>(PassiveEdgeWidth 0 1 2)</tt> \n
 *   The default is zero for all resolutions. A value of 4 will avoid all deformations
 *   at the edge of the image. Make sure that 2*PassiveEdgeWidth < ControlPointGridSize
 *   in each dimension.
 *
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter GridSize: stores the size of the B-spline grid. \n
 *    example: <tt>(GridSize 16 16 16)</tt>
 * \transformparameter GridIndex: stores the index of the B-spline grid. \n
 *    example: <tt>(GridIndex 0 0 0)</tt>
 * \transformparameter GridSpacing: stores the spacing of the B-spline grid. \n
 *    example: <tt>(GridSpacing 16.0 16.0 16.0)</tt>
 * \transformparameter GridOrigin: stores the origin of the B-spline grid. \n
 *    example: <tt>(GridOrigin 0.0 0.0 0.0)</tt>
 * \transformparameter GridDirection: stores the direction cosines of the B-spline grid. \n
 *    example: <tt>(GridDirection 1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 0.1)</tt>
 * \transformparameter BSplineTransformSplineOrder: stores the B-spline order 1,2, or 3. \n
 *    example: <tt>(BSplineTransformSplineOrder 3)</tt>
 *    Default value: 3 (cubic B-splines).
 * \transformparameter StackSpacing: stores the spacing between the sub transforms. \n
 *    exanoke: <tt>(StackSpacing 1.0)</tt>
 * \transformparameter StackOrigin: stores the origin of the first sub transform. \n
 *    exanoke: <tt>(StackOrigin 0.0)</tt>
 * \transformparameter NumberOfSubTransforms: stores the number of sub transforms. \n
 *    exanoke: <tt>(NumberOfSubTransforms 10)</tt>
 *
 * \todo It is unsure what happens when one of the image dimensions has length 1.
 *
 * \ingroup Transforms
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT BSplineStackTransform
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineStackTransform);

  /** Standard ITK-stuff. */
  using Self = BSplineStackTransform;
  using Superclass1 = itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                        elx::TransformBase<TElastix>::FixedImageDimension>;
  using Superclass2 = elx::TransformBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BSplineStackTransform, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "BSplineStackTransform")</tt>\n
   */
  elxClassNameMacro("BSplineStackTransform");

  /** (Reduced) dimension of the fixed image. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);
  itkStaticConstMacro(ReducedSpaceDimension, unsigned int, Superclass2::FixedImageDimension - 1);

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform.
   */
  using BSplineTransformBaseType =
    itk::AdvancedBSplineDeformableTransformBase<typename elx::TransformBase<TElastix>::CoordRepType,
                                                Self::SpaceDimension>;
  using BSplineTransformBasePointer = typename BSplineTransformBaseType::Pointer;

  /** The ITK-class for the sub transforms, which have a reduced dimension. */
  using ReducedDimensionBSplineTransformBaseType =
    itk::AdvancedBSplineDeformableTransformBase<typename elx::TransformBase<TElastix>::CoordRepType,
                                                Self::ReducedSpaceDimension>;
  using ReducedDimensionBSplineTransformBasePointer = typename ReducedDimensionBSplineTransformBaseType::Pointer;

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::ParametersType;
  using typename Superclass2::ParameterMapType;
  using typename Superclass1::NumberOfParametersType;

  /** Typedef's specific for the BSplineTransform. */
  using PixelType = typename BSplineTransformBaseType::PixelType;
  using ImageType = typename BSplineTransformBaseType::ImageType;
  using ImagePointer = typename BSplineTransformBaseType::ImagePointer;
  using RegionType = typename BSplineTransformBaseType::RegionType;
  using IndexType = typename BSplineTransformBaseType::IndexType;
  using SizeType = typename BSplineTransformBaseType::SizeType;
  using SpacingType = typename BSplineTransformBaseType::SpacingType;
  using OriginType = typename BSplineTransformBaseType::OriginType;
  using DirectionType = typename BSplineTransformBaseType::DirectionType;

  /** Typedef's from TransformBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using typename Superclass2::CoordRepType;
  using typename Superclass2::FixedImageType;
  using typename Superclass2::MovingImageType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using CombinationTransformType = typename Superclass2::CombinationTransformType;

  /** Reduced dimension image typedefs. */
  using ReducedDimensionImageType = itk::Image<PixelType, Self::ReducedSpaceDimension>;
  using ReducedDimensionRegionType = itk::ImageRegion<Self::ReducedSpaceDimension>;
  using ReducedDimensionSizeType = typename ReducedDimensionRegionType::SizeType;
  using ReducedDimensionIndexType = typename ReducedDimensionRegionType::IndexType;
  using ReducedDimensionSpacingType = typename ReducedDimensionImageType::SpacingType;
  using ReducedDimensionDirectionType = typename ReducedDimensionImageType::DirectionType;
  using ReducedDimensionOriginType = typename ReducedDimensionImageType::PointType;

  /** Typedef's for the GridScheduleComputer and the UpsampleBSplineParametersFilter. */
  using GridScheduleComputerType = itk::GridScheduleComputer<CoordRepType, ReducedSpaceDimension>;
  using GridScheduleComputerPointer = typename GridScheduleComputerType::Pointer;
  using GridScheduleType = typename GridScheduleComputerType ::VectorGridSpacingFactorType;
  using GridUpsamplerType = itk::UpsampleBSplineParametersFilter<ParametersType, ReducedDimensionImageType>;
  using GridUpsamplerPointer = typename GridUpsamplerType::Pointer;

  /** Typedef's creation of parameter map */
  using ParameterValueType = std::vector<std::string>;

  /** Execute stuff before anything else is done:
   * \li Initialize the right BSplineTransform.
   * \li Initialize the right grid schedule computer.
   * \li Initialize upsample filter.
   */
  int
  BeforeAll() override;

  /** Execute stuff before the actual registration:
   * \li Create an initial B-spline grid.
   * \li Set the stack transform parameters.
   * \li Set initial sub transforms.
   * \li Create initial registration parameters.
   * \li PrecomputeGridInformation.
   * Initially, the transform is set to use a 1x1x1 grid, with deformation (0,0,0).
   * In the method BeforeEachResolution() this will be replaced by the right grid size.
   * This seems not logical, but it is required, since the registration
   * class checks if the number of parameters in the transform is equal to
   * the number of parameters in the registration class. This check is done
   * before calling the BeforeEachResolution() methods.
   */
  void
  BeforeRegistration() override;

  /** Execute stuff before each new pyramid resolution:
   * \li In the first resolution call InitializeTransform().
   * \li In next resolutions upsample the B-spline grid if necessary (so, call IncreaseScale())
   */
  void
  BeforeEachResolution() override;

  /** Method to increase the density of the B-spline grid.
   * \li Determine the new B-spline coefficients that describe the current deformation field
   *     for all sub transforms.
   * \li Set these coefficients as InitialParametersOfNextLevel in the registration object.
   * Called by BeforeEachResolution().
   */
  virtual void
  IncreaseScale();

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile() override;

  /** Set the scales of the edge B-spline coefficients to zero. */
  virtual void
  SetOptimizerScales(const unsigned int edgeWidth);

protected:
  /** The constructor. */
  BSplineStackTransform() { this->Superclass1::SetCurrentTransform(m_StackTransform); }

  /** The destructor. */
  ~BSplineStackTransform() override = default;

  /** Read user-specified gridspacing and call the itkGridScheduleComputer. */
  virtual void
  PreComputeGridInformation();

private:
  elxOverrideGetSelfMacro;

  /** Method to set the initial B-spline grid and grid scheduler and initialize the parameters (to 0).
   * \li Define the initial grid region, origin and spacing, using the precomputed grid information.
   * \li Set the initial parameters to zero and set then as InitialParametersOfNextLevel in the registration object.
   * Called by BeforeEachResolution().
   */
  void
  InitializeTransform();

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap() const override;

  /** The deleted copy constructor and assignment operator. */
  /** Typedef for stack transform. */
  using StackTransformType = itk::BSplineStackTransform<SpaceDimension>;

  /** The B-spline stack transform. */
  const typename StackTransformType::Pointer m_StackTransform{ StackTransformType::New() };

  /** Dummy sub transform to be used to set sub transforms of stack transform. */
  ReducedDimensionBSplineTransformBasePointer m_DummySubTransform;

  /** Grid schedule computer and grid upsampler. */
  GridScheduleComputerPointer m_GridScheduleComputer;
  GridUpsamplerPointer        m_GridUpsampler;

  /** Variable to remember order of B-spline transform. */
  unsigned int m_SplineOrder;

  /** Stack variables. */
  unsigned int m_NumberOfSubTransforms;
  double       m_StackOrigin, m_StackSpacing;

  /** Initialize the right B-spline transform based on the spline order. */
  unsigned int
  InitializeBSplineTransform();
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxBSplineStackTransform.hxx"
#endif

#endif // end #ifndef elxBSplineStackTransform_h
