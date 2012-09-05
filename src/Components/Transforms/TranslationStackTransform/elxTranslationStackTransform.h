/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

/**
 * \class TranslationStackTransform
 * \brief A Translation transform based on the itkStackTransform.
 *
 * This transform is a Translation transformation. Calls to TransformPoint and GetJacobian are
 * redirected to the appropriate sub transform based on the last dimension (time) index.
 *
 * This transform uses the size, spacing and origin of the last dimension of the fixed
 * image to set the number of sub transforms the origin of the first transform and the
 * spacing between the transforms.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "TranslationStackTransform")</tt>
 *
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

#ifndef __elxTranslationStackTransform_h
#define __elxTranslationStackTransform_h

/** Include itk transforms needed. */
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedTranslationTransform.h"
#include "itkStackTransform.h"

#include "elxIncludes.h"


namespace elastix
{
    template < class TElastix >
class TranslationStackTransform
  : public itk::AdvancedCombinationTransform<
        typename elx::TransformBase<TElastix>::CoordRepType,
        elx::TransformBase<TElastix>::FixedImageDimension > ,
  public elx::TransformBase<TElastix>
{
public:

  /** Standard ITK-stuff. */
  typedef TranslationStackTransform                         Self;

  typedef itk::AdvancedCombinationTransform<
    typename elx::TransformBase<TElastix>::CoordRepType,
    elx::TransformBase<TElastix>::FixedImageDimension >     Superclass1;

  typedef elx::TransformBase<TElastix>                      Superclass2;

  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( TranslationStackTransform, itk::AdvancedCombinationTransform );

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "TranslationStackTransform")</tt>\n
   */
  elxClassNameMacro( "TranslationStackTransform" );

  /** (Reduced) dimension of the fixed image. */
  itkStaticConstMacro( SpaceDimension, unsigned int, Superclass2::FixedImageDimension );
  itkStaticConstMacro( ReducedSpaceDimension, unsigned int, Superclass2::FixedImageDimension - 1 );

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform.
   */
  typedef itk::AdvancedTranslationTransform < 
      typename elx::TransformBase < TElastix >::CoordRepType, 
      itkGetStaticConstMacro( SpaceDimension ) > 	             TranslationTransformType;
  typedef typename TranslationTransformType::Pointer            TranslationTransformPointer;

  /** The ITK-class for the sub transforms, which have a reduced dimension. */
  typedef itk::AdvancedTranslationTransform<
    typename elx::TransformBase<TElastix>::CoordRepType,
    itkGetStaticConstMacro( ReducedSpaceDimension ) >                  ReducedDimensionTranslationTransformType;
  typedef typename ReducedDimensionTranslationTransformType::Pointer   ReducedDimensionTranslationTransformPointer;

  /** Typedef for stack transform. */
  typedef itk::StackTransform<
    typename elx::TransformBase<TElastix>::CoordRepType,
    itkGetStaticConstMacro( SpaceDimension ),
    itkGetStaticConstMacro( SpaceDimension ) >                TranslationStackTransformType;
  typedef typename TranslationStackTransformType::Pointer     TranslationStackTransformPointer;

  /** Typedef SizeType. */
  typedef typename FixedImageType::SizeType               SizeType;

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::ParametersType            ParametersType;
  typedef typename Superclass1::NumberOfParametersType    NumberOfParametersType;

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
  typedef typename Superclass2::CombinationTransformType  CombinationTransformType;

   /** Execute stuff before the actual registration:
   * \li Set the stack transform parameters.
   * \li Set initial sub transforms.
   * \li Create initial registration parameters.
   */
  virtual int BeforeAll( void );

  virtual void BeforeRegistration( void );

  virtual void InitializeTransform( void );

  /** Function to read transform-parameters from a file. */
  virtual void ReadFromFile( void );

  /** Function to write transform-parameters to a file. */
  virtual void WriteToFile( const ParametersType & param ) const;


protected:

  /** The constructor. */
  TranslationStackTransform();

  /** The destructor. */
  virtual ~TranslationStackTransform() {}

 
private:

  /** The private constructor and copy constructor. */
  TranslationStackTransform( const Self& );  // purposely not implemented
  void operator=( const Self& );         // purposely not implemented

  /** The Translation stack transform. */
  TranslationStackTransformPointer  m_TranslationStackTransform;

  /** Dummy sub transform to be used to set sub transforms of stack transform. */
  ReducedDimensionTranslationTransformPointer  m_TranslationDummySubTransform;

  /** Stack variables. */
  unsigned int m_NumberOfSubTransforms;
  double m_StackOrigin, m_StackSpacing;

  unsigned int InitializeTranslationTransform();

}; // end class TranslationStackTransform


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxTranslationStackTransform.hxx"
#endif

#endif // end #ifndef __elxTranslationStackTransform_h

