/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkDeformationVectorFieldTransform_H__
#define __itkDeformationVectorFieldTransform_H__

//#include <iostream>
#include "itkBSplineDeformableTransform.h"
//#include "itkImage.h"
//#include "itkVector.h"

namespace itk
{

/** \class DeformationVectorFieldTransform
 * \brief An itk transform based on a DeformationVectorField
 *
 * This class inherits from the BSplineDeformableTransform,
 * but sets the controlpoints at every index point.
 *
 * \ingroup Transforms
 */

	// I take a 0-th order BSplineDeformableTransform, where a
	// 1-st order might be better. However, in that case:
	// TODO: take care of image edges, where there is BSpline support!
	template <
    class TScalarType = double,				// Data type for scalars (float or double)
    unsigned int NDimensions = 3 >		// Number of dimensions
		class DeformationVectorFieldTransform:
	public BSplineDeformableTransform< TScalarType, NDimensions, 0 >
	{
	public:
		
		/** Standard class typedefs. */
		typedef DeformationVectorFieldTransform				Self;
		typedef BSplineDeformableTransform<
			TScalarType, NDimensions, 0 >								Superclass;
		typedef SmartPointer< Self >									Pointer;
		typedef SmartPointer< const Self >						ConstPointer;
		
		/** New macro for creation of through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( DeformationVectorFieldTransform, BSplineDeformableTransform );
		
		/** Dimension of the domain space. */
		itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );
		
		/** Typedef's inherited from Superclass. */
		typedef typename Superclass::ScalarType							ScalarType;
		typedef typename Superclass::ParametersType					ParametersType;
		typedef typename Superclass::JacobianType						JacobianType;
		typedef typename Superclass::InputVectorType				InputVectorType;
		typedef typename Superclass::OutputVectorType				OutputVectorType;
		typedef typename Superclass::InputCovariantVectorType		InputCovariantVectorType;
		typedef typename Superclass::OutputCovariantVectorType	OutputCovariantVectorType;
		typedef typename Superclass::InputVnlVectorType			InputVnlVectorType;
		typedef typename Superclass::OutputVnlVectorType		OutputVnlVectorType;
		typedef typename Superclass::InputPointType					InputPointType;
		typedef typename Superclass::OutputPointType				OutputPointType;

		/** Typedef's for BulkTransform. */
		typedef typename Superclass::BulkTransformType			BulkTransformType;
		typedef typename Superclass::BulkTransformPointer		BulkTransformPointer;

		/** Parameters as SpaceDimension number of images. */
		typedef typename ParametersType::ValueType		PixelType;
		typedef Image< PixelType,
			itkGetStaticConstMacro( SpaceDimension ) >	ImageType;
		typedef typename ImageType::Pointer						ImagePointer;

		/** Typedef's for VectorImage. */
		typedef Vector< float,
			itkGetStaticConstMacro( SpaceDimension ) >	VectorPixelType;
		typedef Image< VectorPixelType,
			itkGetStaticConstMacro( SpaceDimension ) >	VectorImageType;
		typedef typename VectorImageType::Pointer			VectorImagePointer;
		
		/** Set the coefficient image. */
		virtual void SetCoefficientImage( VectorImagePointer vecImage );
		
		/** Don't allow the GridRegion, GridSpacing, and GridOrigin
		 * to be set from outside. These are determined by the region,
		 * spacing and origin of the VectorImage.
		 */

		/** This method specifies the region over which the grid resides. *
		virtual void SetGridRegion( const RegionType& region )
		{
			itkExceptionMacro(<< "Method not applicable for deformation field transform. ");
		}
		
		/** This method specifies the grid spacing or resolution. *
		virtual void SetGridSpacing( const SpacingType& spacing )
		{
			itkExceptionMacro(<< "Method not applicable for deformation field transform. ");
		}
		
		/** This method specifies the grid origin. *
		virtual void SetGridOrigin( const OriginType& origin )
		{
			itkExceptionMacro(<< "Method not applicable for deformation field transform. ");
		}
		*/
		
	protected:

		DeformationVectorFieldTransform();
		virtual ~DeformationVectorFieldTransform();

	private:
		
		DeformationVectorFieldTransform( const Self& );	// purposely not implemented
		void operator=( const Self& );									// purposely not implemented

		/** Member variables. */
		ImagePointer m_Images[ SpaceDimension ];
	
	}; // end class DeformationVectorFieldTransform

} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDeformationVectorFieldTransform.txx"
#endif

#endif // end #ifndef __itkDeformationVectorFieldTransform_H__

