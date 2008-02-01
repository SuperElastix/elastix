#ifndef __elxBSplineTransform_h
#define __elxBSplineTransform_h

/* For easy changing the BSplineOrder: */
#define __VSplineOrder 3

#include "itkBSplineCombinationTransform.h"

#include "itkGridScheduleComputer.h"
#include "itkUpsampleBSplineParametersFilter.h"

#include "elxIncludes.h"


namespace elastix
{
using namespace itk;

	/**
	 * \class BSplineTransform
	 * \brief A transform based on the itkBSplineTransform.
	 *
	 * This transform is a B-spline transformation, commonly used for nonrigid registration.
	 *
	 * The parameters used in this class are:
	 * \parameter Transform: Select this transform as follows:\n
	 *		<tt>(%Transform "BSplineTransform")</tt>  
   * \parameter FinalGridSpacing: DEPRECATED. the grid spacing of the B-spline transform for each dimension. \n
	 *		example: <tt>(FinalGridSpacing 8.0 8.0 8.0)</tt> \n
	 *    If only one argument is given, that factor is used for each dimension. The spacing
	 *    is not in millimeters, but in "voxel size units".
	 *		The default is 16.0 in every dimension. The option is deprecated. Use either FinalGridSpacingInVoxels
   *    or FinalGridSpacingInPhysicalUnits.
   * \parameter FinalGridSpacingInVoxels: the grid spacing of the B-spline transform for each dimension. \n
	 *		example: <tt>(FinalGridSpacingInVoxels 8.0 8.0 8.0)</tt> \n
	 *    If only one argument is given, that factor is used for each dimension. The spacing
	 *    is not in millimeters, but in "voxel size units".
	 *		The default is 16.0 in every dimension.
   * \parameter FinalGridSpacingInPhysicalUnits: the grid spacing of the B-spline transform for each dimension. \n
	 *		example: <tt>(FinalGridSpacingInPhysicalUnits 8.0 8.0 8.0)</tt> \n
	 *    If only one argument is given, that factor is used for each dimension. The spacing
	 *    is specified in millimeters.
	 *		If not specified, the FinalGridSpacingInVoxels is used, or the FinalGridSpacing,
   *    to compute a FinalGridSpacingInPhysicalUnits. If those are not specified, the default 
   *    value for FinalGridSpacingInVoxels is used to compute a FinalGridSpacingInPhysicalUnits.
	 * \parameter UpsampleGridOption: DEPRECATED. a flag to determine if the B-spline grid should
	 *		be upsampled from one resolution level to another. Choose from {true, false}. \n
	 *		example: <tt>(UpsampleGridOption "true")</tt> \n
	 *		example: <tt>(UpsampleGridOption "true" "false" "true")</tt> \n
	 *		The default is "true" inbetween all resolutions. The option is depecrated. Better use
   *    the GridSpacingSchedule.
   * \parameter GridSpacingSchedule: the grid spacing downsampling factors for the B-spline transform
   *    for each dimension and each resolution. \n
	 *		example: <tt>(GridSpacingSchedule 4.0 4.0 2.0 2.0 1.0 1.0)</tt> \n
   *    Which is an example for a 2D image, using 3 resolutions. \n
   *    For convenience, you may also specify only one value for each resolution:\n
   *		example: <tt>(GridSpacingSchedule 4.0 2.0 1.0 )</tt> \n
   *    which is equivalent to the example above.
   * \parameter PassiveEdgeWidth: the width of a band of control points at the border of the
   *   BSpline coefficient image that should remain passive during optimisation. \n
   *   Can be specified for each resolution. \n
   *   example: <tt>(PassiveEdgeWidth 0 1 2)</tt> \n
   *   The default is zero for all resolutions. A value of 4 will avoid all deformations
   *   at the edge of the image. Make sure that 2*PassiveEdgeWidth < ControlPointGridSize
   *   in each dimension.
   *   
	 * 
	 * The transform parameters necessary for transformix, additionally defined by this class, are:
	 * \transformparameter GridSize: stores the size of the B-spline grid. \n
	 *		example: <tt>(GridSize 16 16 16)</tt>
	 * \transformparameter GridIndex: stores the index of the B-spline grid. \n
	 *		example: <tt>(GridIndex 0 0 0)</tt>
	 * \transformparameter GridSpacing: stores the spacing of the B-spline grid. \n
	 *		example: <tt>(GridSpacing 16.0 16.0 16.0)</tt>
	 * \transformparameter GridOrigin: stores the origin of the B-spline grid. \n
	 *		example: <tt>(GridOrigin 0.0 0.0 0.0)</tt>
	 *
	 * \todo It is unsure what happens when one of the image dimensions has length 1. 
	 * 
	 * \ingroup Transforms
	 */

	template < class TElastix >
		class BSplineTransform:
	public
		BSplineCombinationTransform< 
			ITK_TYPENAME elx::TransformBase<TElastix>::CoordRepType,
			elx::TransformBase<TElastix>::FixedImageDimension,
			__VSplineOrder > ,
	public
		TransformBase<TElastix>
	{
	public:

		/** Standard ITK-stuff. */
		typedef BSplineTransform 										    Self;
		typedef BSplineCombinationTransform<
			typename elx::TransformBase<TElastix>::CoordRepType,
			elx::TransformBase<TElastix>::FixedImageDimension,
			__VSplineOrder >													    Superclass1;
		typedef elx::TransformBase<TElastix>				    Superclass2;
    typedef SmartPointer<Self>									    Pointer;
		typedef SmartPointer<const Self>						    ConstPointer;
				
		/** The ITK-class that provides most of the functionality, and
		 * that is set as the "CurrentTransform" in the CombinationTransform.
     */
		typedef typename 
			Superclass1::BSplineTransformType					    BSplineTransformType;
    typedef typename BSplineTransformType::Pointer	BSplineTransformPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( BSplineTransform, BSplineCombinationTransform );

		/** Name of this class.
		 * Use this name in the parameter file to select this specific transform. \n
		 * example: <tt>(Transform "BSplineTransform")</tt>\n
		 */
		elxClassNameMacro( "BSplineTransform" );
		
		/** Dimension of the fixed image. */
		itkStaticConstMacro( SpaceDimension, unsigned int, Superclass2::FixedImageDimension );
		
		/** The BSpline order. */
		itkStaticConstMacro( SplineOrder, unsigned int, __VSplineOrder );
		
		/** Typedefs inherited from the superclass. */
		typedef typename Superclass1::ScalarType 								ScalarType;
		typedef typename Superclass1::ParametersType 						ParametersType;
		typedef typename Superclass1::JacobianType 							JacobianType;
		typedef typename Superclass1::InputVectorType						InputVectorType;
		typedef typename Superclass1::OutputVectorType 					OutputVectorType;
		typedef typename Superclass1::InputCovariantVectorType 	InputCovariantVectorType;
		typedef typename Superclass1::OutputCovariantVectorType	OutputCovariantVectorType;
		typedef typename Superclass1::InputVnlVectorType 				InputVnlVectorType;
		typedef typename Superclass1::OutputVnlVectorType				OutputVnlVectorType;
		typedef typename Superclass1::InputPointType 						InputPointType;
		typedef typename Superclass1::OutputPointType						OutputPointType;
		
		/** Typedef's specific for the BSplineTransform. */
		typedef typename BSplineTransformType::PixelType				PixelType;
		typedef typename BSplineTransformType::ImageType				ImageType;
		typedef typename BSplineTransformType::ImagePointer			ImagePointer;
		typedef typename BSplineTransformType::RegionType				RegionType;
		typedef typename BSplineTransformType::IndexType				IndexType;
		typedef typename BSplineTransformType::SizeType					SizeType;
		typedef typename BSplineTransformType::SpacingType			SpacingType;
		typedef typename BSplineTransformType::OriginType				OriginType;
		typedef typename 
			BSplineTransformType::BulkTransformType								BulkTransformType;
		typedef typename 
			BSplineTransformType::BulkTransformPointer						BulkTransformPointer;
		typedef typename 
			BSplineTransformType::WeightsFunctionType							WeightsFunctionType;
		typedef typename BSplineTransformType::WeightsType			WeightsType;
		typedef typename 
			BSplineTransformType::ContinuousIndexType							ContinuousIndexType;
		typedef typename 
			BSplineTransformType::ParameterIndexArrayType					ParameterIndexArrayType;

		/** Typedef's from TransformBase. */
		typedef typename Superclass2::ElastixType								ElastixType;
		typedef typename Superclass2::ElastixPointer						ElastixPointer;
		typedef typename Superclass2::ConfigurationType					ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer			ConfigurationPointer;
		typedef typename Superclass2::RegistrationType					RegistrationType;
		typedef typename Superclass2::RegistrationPointer				RegistrationPointer;
		typedef typename Superclass2::CoordRepType							CoordRepType;
		typedef typename Superclass2::FixedImageType						FixedImageType;
		typedef typename Superclass2::MovingImageType						MovingImageType;
		typedef typename Superclass2::ITKBaseType								ITKBaseType;
		typedef typename Superclass2::CombinationTransformType	CombinationTransformType;

		/** Typedef's for the GridScheduleComputer and the UpsampleBSplineParametersFilter. */
    typedef GridScheduleComputer<
      CoordRepType, SpaceDimension >                        GridScheduleComputerType;
    typedef typename GridScheduleComputerType::Pointer      GridScheduleComputerPointer;
    typedef typename GridScheduleComputerType
      ::VectorGridSpacingFactorType                         GridScheduleType;
    typedef UpsampleBSplineParametersFilter<
      ParametersType, ImageType >                           GridUpsamplerType;
    typedef typename GridUpsamplerType::Pointer             GridUpsamplerPointer;

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
		virtual void BeforeRegistration( void );

		/** Execute stuff before each new pyramid resolution:
		 * \li In the first resolution call InitializeTransform().
		 * \li In next resolutions upsample the B-spline grid if necessary (so, call IncreaseScale())
		 */
		virtual void BeforeEachResolution( void );
		
		/** Method to set the initial BSpline grid and initialize the parameters (to 0).
		 * \li Define the initial grid region, origin and spacing, using the precomputed grid information.
		 * \li Set the initial parameters to zero and set then as InitialParametersOfNextLevel in the registration object.
		 * Called by BeforeEachResolution().
		 */
		virtual void InitializeTransform( void );

		/** Method to increase the density of the BSpline grid.
		 * \li Determine the new B-spline coefficients that describe the current deformation field.
		 * \li Set these coefficients as InitialParametersOfNextLevel in the registration object.
		 * Called by BeforeEachResolution().
		 */
		virtual void IncreaseScale( void );		
		
		/** Function to read transform-parameters from a file. */
		virtual void ReadFromFile( void );

		/** Function to write transform-parameters to a file. */
		virtual void WriteToFile( const ParametersType & param );

    /** Set the scales of the edge bspline coefficients to zero. */
    virtual void SetOptimizerScales( unsigned int edgeWidth );
		
	protected:

		/** The constructor. */
		BSplineTransform();

		/** The destructor. */
		virtual ~BSplineTransform() {}

    /** Read user-specified gridspacing and call the itkGridScheduleComputer. */
    virtual void PreComputeGridInformation( void );

    /** Method to determine a schedule based on the UpsampleGridOption */
		virtual void ComputeGridSchedule_Deprecated( GridScheduleType & schedule );


	private:

		/** The private constructor. */
		BSplineTransform( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );		// purposely not implemented


    /** Private variables. */
		BSplineTransformPointer	    m_BSplineTransform;
    GridScheduleComputerPointer m_GridScheduleComputer;
    GridUpsamplerPointer        m_GridUpsampler;
		
	}; // end class BSplineTransform
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxBSplineTransform.hxx"
#endif

#endif // end #ifndef __elxBSplineTransform_h

