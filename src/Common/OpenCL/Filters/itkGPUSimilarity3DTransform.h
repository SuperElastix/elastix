/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUSimilarity3DTransform_h
#define __itkGPUSimilarity3DTransform_h

#include "itkSimilarity3DTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUSimilarity3DTransform
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template< class TScalarType = float,
          class TParentImageFilter = Similarity3DTransform< TScalarType > >
class GPUSimilarity3DTransform :
  public TParentImageFilter,
  public GPUMatrixOffsetTransformBase< TScalarType, 3, 3 >
{
public:
  /** Standard class typedefs. */
  typedef GPUSimilarity3DTransform                          Self;
  typedef TParentImageFilter                                CPUSuperclass;
  typedef GPUMatrixOffsetTransformBase< TScalarType, 3, 3 > GPUSuperclass;
  typedef SmartPointer< Self >                              Pointer;
  typedef SmartPointer< const Self >                        ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUSimilarity3DTransform, CPUSuperclass );

  /** Typedefs */
  typedef typename GPUSuperclass::CPUMatrixType        CPUMatrixType;
  typedef typename GPUSuperclass::CPUInverseMatrixType CPUInverseMatrixType;
  typedef typename GPUSuperclass::CPUOutputVectorType  CPUOutputVectorType;

  /**  */
  virtual const CPUMatrixType &        GetCPUMatrix( void ) const { return this->GetMatrix(); }
  virtual const CPUInverseMatrixType & GetCPUInverseMatrix( void ) const { return this->GetInverseMatrix(); }
  virtual const CPUOutputVectorType &  GetCPUOffset( void ) const { return this->GetOffset(); }

protected:
  GPUSimilarity3DTransform() {}
  virtual ~GPUSimilarity3DTransform() {}

private:
  GPUSimilarity3DTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );         // purposely not implemented
};

/** \class GPUSimilarity3DTransformFactory
* \brief Object Factory implementation for GPUSimilarity3DTransform
*/
class GPUSimilarity3DTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUSimilarity3DTransformFactory Self;
  typedef ObjectFactoryBase               Superclass;
  typedef SmartPointer< Self >            Pointer;
  typedef SmartPointer< const Self >      ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUSimilarity3DTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUSimilarity3DTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUSimilarity3DTransformFactory::Pointer factory =
      GPUSimilarity3DTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUSimilarity3DTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                  // purposely not implemented

#define OverrideSimilarity3DTransformTypeMacro( st )                   \
  {                                                                    \
    this->RegisterOverride(                                            \
      typeid( Similarity3DTransform< st > ).name(),                    \
      typeid( GPUSimilarity3DTransform< st > ).name(),                 \
      "GPU Similarity3DTransform Override",                            \
      true,                                                            \
      CreateObjectFunction< GPUSimilarity3DTransform< st > >::New() ); \
  }

  GPUSimilarity3DTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideSimilarity3DTransformTypeMacro( float );
      //OverrideSimilarity3DTransformTypeMacro(double);

      OverrideSimilarity3DTransformTypeMacro( float );
      //OverrideSimilarity3DTransformTypeMacro(double);

      OverrideSimilarity3DTransformTypeMacro( float );
      //OverrideSimilarity3DTransformTypeMacro(double);
    }
  }
};
} // end namespace itk

#endif /* __itkGPUSimilarity3DTransform_h */
