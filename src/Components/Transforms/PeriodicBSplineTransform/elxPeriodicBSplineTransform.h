/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxPeriodicBSplineTransform_h
#define __elxPeriodicBSplineTransform_h

/* For easy changing the BSplineOrder: */
#define __VSplineOrder 3

#include "itkPeriodicBSplineDeformableTransform.h"

#include "itkPeriodicGridScheduleComputer.h"
#include "itkUpsampleBSplineParametersFilter.h"

#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

  /**
   * \class PeriodicBSplineTransform
   * \brief A transform based on the itkPeriodicBSplineTransform.
   * \ingroup Transforms
   */

  template < class TElastix >
  class PeriodicBSplineTransform :
    public AdvancedCombinationTransform<
    ITK_TYPENAME elx::TransformBase<TElastix>::CoordRepType,
    elx::TransformBase<TElastix>::FixedImageDimension > ,
    public 
      elx::TransformBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef PeriodicBSplineTransform                  Self;

    typedef AdvancedCombinationTransform<
      typename elx::TransformBase<TElastix>::CoordRepType,
      elx::TransformBase<TElastix>::FixedImageDimension > 
                                                    Superclass1;
    typedef elx::TransformBase<TElastix>            Superclass2;
    typedef SmartPointer<Self>                      Pointer;
    typedef SmartPointer<const Self>                ConstPointer;
        
    /** The ITK-class that provides most of the functionality, and
    * that is set as the "CurrentTransform" in the CombinationTransform */
    typedef PeriodicBSplineDeformableTransform<
      typename elx::TransformBase<TElastix>::CoordRepType,
      elx::TransformBase<TElastix>::FixedImageDimension,
      __VSplineOrder >         PeriodicBSplineTransformType;

    typedef typename PeriodicBSplineTransformType::Pointer 
                                                PeriodicBSplineTransformPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( PeriodicBSplineTransform, AdvancedCombinationTransform );

    /** Name of this class.
     * Use this name in the parameter file to select this specific transform. \n
     * example: <tt>(Transform "BSplineTransform")</tt>\n
     */
    elxClassNameMacro( "PeriodicBSplineTransform" );

    /** Dimension of the fixed image. */
    itkStaticConstMacro( SpaceDimension, unsigned int, 
                                            Superclass2::FixedImageDimension );

    /** The BSpline order. */
    itkStaticConstMacro( SplineOrder, unsigned int, __VSplineOrder );
    
    /** Typedefs inherited from the superclass. */
    typedef typename Superclass1::ScalarType          ScalarType;
    typedef typename Superclass1::ParametersType      ParametersType;
    typedef typename Superclass1::JacobianType        JacobianType;
    typedef typename Superclass1::InputVectorType     InputVectorType;
    typedef typename Superclass1::OutputVectorType    OutputVectorType;
    typedef typename Superclass1::InputCovariantVectorType  
                                                      InputCovariantVectorType;
    typedef typename Superclass1::OutputCovariantVectorType 
                                                      OutputCovariantVectorType;
    typedef typename Superclass1::InputVnlVectorType  InputVnlVectorType;
    typedef typename Superclass1::OutputVnlVectorType OutputVnlVectorType;
    typedef typename Superclass1::InputPointType      InputPointType;
    typedef typename Superclass1::OutputPointType     OutputPointType;
    
    /** Typedef's specific for the BSplineTransform. */
    typedef typename PeriodicBSplineTransformType::PixelType    PixelType;
    typedef typename PeriodicBSplineTransformType::ImageType    ImageType;
    typedef typename PeriodicBSplineTransformType::ImagePointer ImagePointer;
    typedef typename PeriodicBSplineTransformType::RegionType   RegionType;
    typedef typename PeriodicBSplineTransformType::IndexType    IndexType;
    typedef typename PeriodicBSplineTransformType::SizeType     SizeType;
    typedef typename PeriodicBSplineTransformType::SpacingType  SpacingType;
    typedef typename PeriodicBSplineTransformType::OriginType   OriginType;
    typedef typename PeriodicBSplineTransformType::DirectionType DirectionType;
    typedef typename PeriodicBSplineTransformType::BulkTransformType
                                                        BulkTransformType;
    typedef typename PeriodicBSplineTransformType::BulkTransformPointer     
                                                        BulkTransformPointer;
    typedef typename PeriodicBSplineTransformType::WeightsFunctionType      
                                                        WeightsFunctionType;
    typedef typename PeriodicBSplineTransformType::WeightsType  WeightsType;
    typedef typename PeriodicBSplineTransformType::ContinuousIndexType 
                                                        ContinuousIndexType;
    typedef typename PeriodicBSplineTransformType::ParameterIndexArrayType  
                                                        ParameterIndexArrayType;

    /** Typedef's from TransformBase. */
    typedef typename Superclass2::ElastixType               ElastixType;
    typedef typename Superclass2::ElastixPointer            ElastixPointer;
    typedef typename Superclass2::ConfigurationType         ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer      ConfigurationPointer;
    typedef typename Superclass2::RegistrationType          RegistrationType;
    typedef typename Superclass2::RegistrationPointer       RegistrationPointer;
    typedef typename Superclass2::CoordRepType              CoordRepType;
    typedef typename Superclass2::FixedImageType            FixedImageType;
    typedef typename Superclass2::MovingImageType           MovingImageType;
    typedef typename Superclass2::ITKBaseType               ITKBaseType;
    typedef typename Superclass2::CombinationTransformType  
                                                      CombinationTransformType;

    /** Typedef's for the GridScheduleComputer and the UpsampleBSplineParametersFilter. */
    typedef PeriodicGridScheduleComputer<CoordRepType, SpaceDimension >             
                                                   GridScheduleComputerType;
    typedef typename GridScheduleComputerType::Pointer     
                                                   GridScheduleComputerPointer;
    typedef typename GridScheduleComputerType::VectorGridSpacingFactorType  
                                                   GridScheduleType;
    typedef UpsampleBSplineParametersFilter<ParametersType, ImageType >
                                                   GridUpsamplerType;
    typedef typename GridUpsamplerType::Pointer                             
                                                   GridUpsamplerPointer;

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
    virtual void WriteToFile( const ParametersType & param ) const;

    /** Set the scales of the edge B-spline coefficients to zero. */
    virtual void SetOptimizerScales( unsigned int edgeWidth );
    
  protected:

    /** The constructor. */
    PeriodicBSplineTransform();

    /** The destructor. */
    virtual ~PeriodicBSplineTransform() {}

    /** Read user-specified gridspacing and call the itkGridScheduleComputer. */
    virtual void PreComputeGridInformation( void );
 
  private:

    /** The private constructor. */
    PeriodicBSplineTransform( const Self& );  // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );    // purposely not implemented


    /** Private variables. */
    PeriodicBSplineTransformPointer  m_PeriodicBSplineTransform;
    GridScheduleComputerPointer    m_GridScheduleComputer;
    GridUpsamplerPointer           m_GridUpsampler;
    
  }; // end class BSplineTransform
  

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxPeriodicBSplineTransform.hxx"
#endif

#endif // end #ifndef __elxPeriodicBSplineTransform_h

