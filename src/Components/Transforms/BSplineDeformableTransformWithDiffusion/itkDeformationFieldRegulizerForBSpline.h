/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkDeformationFieldRegulizerForBSpline_H__
#define __itkDeformationFieldRegulizerForBSpline_H__

#include "itkDeformationFieldRegulizer.h"


namespace itk
{
    
  /**
   * \class DeformationFieldRegulizerForBSpline
   * \brief This class combines a B-spline transform with the
   * DeformationFieldTransform.
   *
   * This class inherits from the DeformationFieldRegulizer and only
   * overwrites the TransformPoint() function. This is necessary for
   * the Mattes MI metric. This class is templated over TAnyITKTransform,
   * but it is in fact only the BSplineDeformableTransform.
   *
   * \ingroup Transforms
   * \ingroup Common
   */
  
  template < class TBSplineTransform >
  class DeformationFieldRegulizerForBSpline :
    public DeformationFieldRegulizer<
    TBSplineTransform >
  {
  public:
    
    /** Standard itk. */
    typedef DeformationFieldRegulizerForBSpline     Self;
    typedef DeformationFieldRegulizer<
      TBSplineTransform >                           Superclass;
    typedef SmartPointer< Self >                    Pointer;
    typedef SmartPointer< const Self >              ConstPointer;
    
    /** Method for creation through the object factory. */
    itkNewMacro( Self );
    
    /** Run-time type information (and related methods). */
    itkTypeMacro( DeformationFieldRegulizerForBSpline, DeformationFieldRegulizer );
    
    /** Get the dimension of the input space. */
    itkStaticConstMacro( InputSpaceDimension, unsigned int, Superclass::InputSpaceDimension );
    /** Get the dimension of the output space. */
    itkStaticConstMacro( OutputSpaceDimension, unsigned int, Superclass::OutputSpaceDimension );
    
    /** Typedef's inherited from Superclass. */
    typedef typename Superclass::ScalarType                 ScalarType;
    typedef typename Superclass::ParametersType             ParametersType;
    typedef typename Superclass::JacobianType               JacobianType;
    typedef typename Superclass::InputVectorType            InputVectorType;
    typedef typename Superclass::OutputVectorType           OutputVectorType;
    typedef typename Superclass::InputCovariantVectorType   InputCovariantVectorType;
    typedef typename Superclass::OutputCovariantVectorType  OutputCovariantVectorType;
    typedef typename Superclass::InputVnlVectorType         InputVnlVectorType;
    typedef typename Superclass::OutputVnlVectorType        OutputVnlVectorType;
    typedef typename Superclass::InputPointType             InputPointType;
    typedef typename Superclass::OutputPointType            OutputPointType;

    /** Typedef's inherited from the BSplineTransform. */
    typedef TBSplineTransform     BSplineTransformType;
    typedef typename BSplineTransformType::WeightsType                WeightsType;
    typedef typename BSplineTransformType::ParameterIndexArrayType    ParameterIndexArrayType;

    /** Other typedef's inherited from Superclass. */
    typedef typename Superclass::IntermediaryDFTransformType          IntermediaryDFTransformType;
    typedef typename Superclass::VectorImageType                      VectorImageType;

    /** Method to transform a point, 1 argument. */
    virtual OutputPointType TransformPoint( const InputPointType & point ) const;

    /**  Method to transform a point, 5 arguments. */
    virtual void TransformPoint(
      const InputPointType &inputPoint,
      OutputPointType &outputPoint,
      WeightsType &weights,
      ParameterIndexArrayType &indices,
      bool &inside ) const;
    
  protected:
    
    /** The constructor. */
    DeformationFieldRegulizerForBSpline();
    /** The destructor. */
    virtual ~DeformationFieldRegulizerForBSpline() {};
    
  private:
    
    /** The private constructor. */
    DeformationFieldRegulizerForBSpline( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );                      // purposely not implemented
    
  }; // end class DeformationFieldRegulizerForBSpline
    
    
} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDeformationFieldRegulizerForBSpline.hxx"
#endif

#endif // end #ifndef __itkDeformationFieldRegulizerForBSpline_H__

