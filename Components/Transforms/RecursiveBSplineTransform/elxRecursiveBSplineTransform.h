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
#ifndef elxRecursiveBSplineTransform_h
#define elxRecursiveBSplineTransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedCombinationTransform.h"
#include "itkRecursiveBSplineTransform.h"

#include "itkGridScheduleComputer.h"
#include "itkCyclicBSplineDeformableTransform.h"
#include "itkCyclicGridScheduleComputer.h"
#include "itkUpsampleBSplineParametersFilter.h"

namespace elastix
{
/**
 * \class RecursiveBSplineTransform
 * \brief A transform based on the itkRecursiveBSplineTransform.
 *
 * This transform is a B-spline transformation, commonly used for nonrigid registration.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "RecursiveBSplineTransform")</tt>
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
 * \parameter UseCyclicTransform: use the cyclic version of the B-spline transform which
 *   ensures that the B-spline polynomials wrap around in the slowest varying dimension.
 *   This is useful for dynamic imaging data in which the motion is assumed to be cyclic,
 *   for example in ECG-gated or respiratory gated CTA. For more information see the paper:
 *   <em>Nonrigid registration of dynamic medical imaging data using nD+t B-splines and a
 *   groupwise optimization approach</em>, C.T. Metz, S. Klein, M. Schaap, T. van Walsum and
 *   W.J. Niessen, Medical Image Analysis, in press.
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
 * \transformparameter UseCyclicTransform: use the cyclic version of the B-spline transform which
 *   ensures that the B-spline polynomials wrap around in the slowest varying dimension.
 *   This is useful for dynamic imaging data in which the motion is assumed to be cyclic,
 *   for example in ECG-gated or respiratory gated CTA. For more information see the paper:
 *   <em>Nonrigid registration of dynamic medical imaging data using nD+t B-splines and a
 *   groupwise optimization approach</em>, C.T. Metz, S. Klein, M. Schaap, T. van Walsum and
 *   W.J. Niessen, Medical Image Analysis, in press.
 *
 * \todo It is unsure what happens when one of the image dimensions has length 1.
 *
 * \ingroup Transforms
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT RecursiveBSplineTransform
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public TransformBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef RecursiveBSplineTransform Self;
  typedef itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension>
                                        Superclass1;
  typedef elx::TransformBase<TElastix>  Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RecursiveBSplineTransform, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "RecursiveBSplineTransform")</tt>\n
   */
  elxClassNameMacro("RecursiveBSplineTransform");

  /** Dimension of the fixed image. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform.
   */
  typedef itk::AdvancedBSplineDeformableTransformBase<typename elx::TransformBase<TElastix>::CoordRepType,
                                                      itkGetStaticConstMacro(SpaceDimension)>
                                                     BSplineTransformBaseType;
  typedef typename BSplineTransformBaseType::Pointer BSplineTransformBasePointer;

  /** Typedef for supported BSplineTransform types. */
  typedef itk::RecursiveBSplineTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                         itkGetStaticConstMacro(SpaceDimension),
                                         1>
    BSplineTransformLinearType;
  typedef itk::RecursiveBSplineTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                         itkGetStaticConstMacro(SpaceDimension),
                                         2>
    BSplineTransformQuadraticType;
  typedef itk::RecursiveBSplineTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                         itkGetStaticConstMacro(SpaceDimension),
                                         3>
    BSplineTransformCubicType;
  typedef itk::CyclicBSplineDeformableTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                itkGetStaticConstMacro(SpaceDimension),
                                                1>
    CyclicBSplineTransformLinearType;
  typedef itk::CyclicBSplineDeformableTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                itkGetStaticConstMacro(SpaceDimension),
                                                2>
    CyclicBSplineTransformQuadraticType;
  typedef itk::CyclicBSplineDeformableTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                itkGetStaticConstMacro(SpaceDimension),
                                                3>
    CyclicBSplineTransformCubicType;

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::ScalarType                ScalarType;
  typedef typename Superclass1::ParametersType            ParametersType;
  typedef typename Superclass1::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass1::JacobianType              JacobianType;
  typedef typename Superclass1::InputVectorType           InputVectorType;
  typedef typename Superclass1::OutputVectorType          OutputVectorType;
  typedef typename Superclass1::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass1::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass1::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass1::OutputVnlVectorType       OutputVnlVectorType;
  typedef typename Superclass1::InputPointType            InputPointType;
  typedef typename Superclass1::OutputPointType           OutputPointType;

  /** Typedef's specific for the BSplineTransform. */
  typedef typename BSplineTransformBaseType::PixelType               PixelType;
  typedef typename BSplineTransformBaseType::ImageType               ImageType;
  typedef typename BSplineTransformBaseType::ImagePointer            ImagePointer;
  typedef typename BSplineTransformBaseType::RegionType              RegionType;
  typedef typename BSplineTransformBaseType::IndexType               IndexType;
  typedef typename BSplineTransformBaseType::SizeType                SizeType;
  typedef typename BSplineTransformBaseType::SpacingType             SpacingType;
  typedef typename BSplineTransformBaseType::OriginType              OriginType;
  typedef typename BSplineTransformBaseType::DirectionType           DirectionType;
  typedef typename BSplineTransformBaseType::ContinuousIndexType     ContinuousIndexType;
  typedef typename BSplineTransformBaseType::ParameterIndexArrayType ParameterIndexArrayType;

  /** Typedef's from TransformBase. */
  typedef typename Superclass2::ElastixType              ElastixType;
  typedef typename Superclass2::ElastixPointer           ElastixPointer;
  typedef typename Superclass2::ConfigurationType        ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer     ConfigurationPointer;
  typedef typename Superclass2::RegistrationType         RegistrationType;
  typedef typename Superclass2::RegistrationPointer      RegistrationPointer;
  typedef typename Superclass2::CoordRepType             CoordRepType;
  typedef typename Superclass2::FixedImageType           FixedImageType;
  typedef typename Superclass2::MovingImageType          MovingImageType;
  typedef typename Superclass2::ITKBaseType              ITKBaseType;
  typedef typename Superclass2::CombinationTransformType CombinationTransformType;

  /** Typedef's for the GridScheduleComputer and the UpsampleBSplineParametersFilter. */
  typedef itk::GridScheduleComputer<CoordRepType, SpaceDimension>         GridScheduleComputerType;
  typedef itk::CyclicGridScheduleComputer<CoordRepType, SpaceDimension>   CyclicGridScheduleComputerType;
  typedef typename GridScheduleComputerType::Pointer                      GridScheduleComputerPointer;
  typedef typename GridScheduleComputerType ::VectorGridSpacingFactorType GridScheduleType;
  typedef itk::UpsampleBSplineParametersFilter<ParametersType, ImageType> GridUpsamplerType;
  typedef typename GridUpsamplerType::Pointer                             GridUpsamplerPointer;

  /** Typedef that is used in the elastix dll version. */
  typedef typename Superclass2::ParameterMapType ParameterMapType;

  /** Execute stuff before anything else is done:
   * \li Initialize the right BSplineTransform.
   * \li Initialize the right grid schedule computer.
   */
  int
  BeforeAll(void) override;

  /** Execute stuff before the actual registration:
   * \li Create an initial B-spline grid.
   * \li Create initial registration parameters.
   * \li PrecomputeGridInformation
   * Initially, the transform is set to use a 1x1x1 grid, with deformation (0,0,0).
   * In the method BeforeEachResolution() this will be replaced by the right grid size.
   * This seems not logical, but it is required, since the registration
   * class checks if the number of parameters in the transform is equal to
   * the number of parameters in the registration class. This check is done
   * before calling the BeforeEachResolution() methods.
   */
  void
  BeforeRegistration(void) override;

  /** Execute stuff before each new pyramid resolution:
   * \li In the first resolution call InitializeTransform().
   * \li In next resolutions upsample the B-spline grid if necessary (so, call IncreaseScale())
   */
  void
  BeforeEachResolution(void) override;

  /** Method to set the initial B-spline grid and initialize the parameters (to 0).
   * \li Define the initial grid region, origin and spacing, using the precomputed grid information.
   * \li Set the initial parameters to zero and set then as InitialParametersOfNextLevel in the registration object.
   * Called by BeforeEachResolution().
   */
  virtual void
  InitializeTransform(void);

  /** Method to increase the density of the B-spline grid.
   * \li Determine the new B-spline coefficients that describe the current deformation field.
   * \li Set these coefficients as InitialParametersOfNextLevel in the registration object.
   * Called by BeforeEachResolution().
   */
  virtual void
  IncreaseScale(void);

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile(void) override;

  /** Set the scales of the edge B-spline coefficients to zero. */
  virtual void
  SetOptimizerScales(const unsigned int edgeWidth);

protected:
  /** The constructor. */
  RecursiveBSplineTransform();

  /** The destructor. */
  ~RecursiveBSplineTransform() override = default;

  /** Read user-specified grid spacing and call the itkGridScheduleComputer. */
  virtual void
  PreComputeGridInformation(void);

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap(void) const override;

  /** The deleted copy constructor. */
  RecursiveBSplineTransform(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;

  /** Private variables. */
  BSplineTransformBasePointer m_BSplineTransform;
  GridScheduleComputerPointer m_GridScheduleComputer;
  GridUpsamplerPointer        m_GridUpsampler;

  /** Variables to remember order and periodicity of B-spline transform. */
  unsigned int m_SplineOrder;
  bool         m_Cyclic;

  /** Initialize the right B-spline transform based on the spline order and periodicity. */
  unsigned int
  InitializeBSplineTransform();
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxRecursiveBSplineTransform.hxx"
#endif

#endif // end #ifndef elxRecursiveBSplineTransform_h
