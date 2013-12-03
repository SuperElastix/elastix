/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkDeformationVectorFieldTransform_h__
#define __itkDeformationVectorFieldTransform_h__

#include "itkAdvancedBSplineDeformableTransform.h"

namespace itk
{

/** \class DeformationVectorFieldTransform
 * \brief An itk transform based on a DeformationVectorField
 *
 * This class makes it easy to set a deformation vector field
 * as a Transform-object.
 *
 * The class inherits from the 0th-order AdvancedBSplineDeformableTransform,
 * and converts a VectorImage to the B-spline CoefficientImage.
 *
 * This is useful if you know for example how to deform each voxel
 * in an image and want to apply it to that image.
 *
 * \ingroup Transforms
 *
 * \note Better use the DeformationFieldInterpolatingTransform.
 * It is more flexible, since it allows runtime specification of
 * the spline order.
 */

template < class TScalarType = double, unsigned int NDimensions = 3 >
class DeformationVectorFieldTransform
  : public AdvancedBSplineDeformableTransform< TScalarType, NDimensions, 0 >
{
  public:

    /** Standard class typedefs. */
    typedef DeformationVectorFieldTransform       Self;
    typedef AdvancedBSplineDeformableTransform<
      TScalarType, NDimensions, 0 >               Superclass;
    typedef SmartPointer< Self >                  Pointer;
    typedef SmartPointer< const Self >            ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( DeformationVectorFieldTransform, AdvancedBSplineDeformableTransform );

    /** Dimension of the domain space. */
    itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );
    itkStaticConstMacro( SplineOrder, unsigned int, Superclass::SplineOrder );

    /** Typedef's inherited from Superclass. */
    typedef typename Superclass::ScalarType             ScalarType;
    typedef typename Superclass::ParametersType         ParametersType;
    typedef typename Superclass::JacobianType           JacobianType;
    typedef typename Superclass::InputVectorType        InputVectorType;
    typedef typename Superclass::OutputVectorType       OutputVectorType;
    typedef typename Superclass::InputCovariantVectorType   InputCovariantVectorType;
    typedef typename Superclass::OutputCovariantVectorType  OutputCovariantVectorType;
    typedef typename Superclass::InputVnlVectorType     InputVnlVectorType;
    typedef typename Superclass::OutputVnlVectorType    OutputVnlVectorType;
    typedef typename Superclass::InputPointType         InputPointType;
    typedef typename Superclass::OutputPointType        OutputPointType;

    /** Parameters as SpaceDimension number of images. */
    typedef typename Superclass::PixelType              CoefficientPixelType;
    typedef typename Superclass::ImageType              CoefficientImageType;
    typedef typename Superclass::ImagePointer           CoefficientImagePointer;

    /** Typedef's for VectorImage. */
    typedef Vector< float,
      itkGetStaticConstMacro( SpaceDimension ) >          CoefficientVectorPixelType;
    typedef Image< CoefficientVectorPixelType,
      itkGetStaticConstMacro( SpaceDimension ) >          CoefficientVectorImageType;
    typedef typename CoefficientVectorImageType::Pointer  CoefficientVectorImagePointer;

    /** Set the coefficient image as a deformation field.
     * The superclass provides a similar function (SetCoeffficientImage),
     * but this function expects an array of nr_of_dim scalar images.
     * The SetCoefficientVectorImage method accepts a VectorImage,
     * which is often more convenient.
     * The method internally just converts this vector image to
     * nr_of_dim scalar images and passes it on to the
     * SetCoefficientImage function.
     */
    virtual void SetCoefficientVectorImage( const CoefficientVectorImageType * vecImage );

    /** Get the coefficient image as a vector image.
     * The vector image is created only on demand. The caller is
     * expected to provide a smart pointer to the resulting image;
     * this stresses the fact that this method does not return a member
     * variable, like most Get... methods.
     */
    virtual void GetCoefficientVectorImage( CoefficientVectorImagePointer & vecImage ) const;

  protected:

    /** The constructor. */
    DeformationVectorFieldTransform();
    /** The destructor. */
    virtual ~DeformationVectorFieldTransform();

  private:

    /** The private constructor. */
    DeformationVectorFieldTransform( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );                  // purposely not implemented

    /** Member variables. */
    CoefficientImagePointer m_Images[ SpaceDimension ];

  };

} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDeformationVectorFieldTransform.hxx"
#endif

#endif // end #ifndef __itkDeformationVectorFieldTransform_h__
