#ifndef __itkNDImageTemplate_hxx
#define __itkNDImageTemplate_hxx

#include "itkNDImageTemplate.h"

namespace itk
{
	
	/** Constructor */
	template < class TPixel, unsigned int VDimension >
		NDImageTemplate<TPixel, VDimension>::
		NDImageTemplate() 
	{
		m_Image = 0;
		m_Writer = 0;
		m_Reader = 0;
	}
	
	
	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		SetRegions(SizeType size)
	{
		m_Image->SetRegions(
			ConvertToStaticArray<SizeType,SizeTypeD>::DO(size)	 );
	}

		
	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		SetRequestedRegion(DataObject *data)
	{
		m_Image->SetRequestedRegion(data);
	}
	
	
	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		Allocate(void)
	{
		m_Image->Allocate();
	}
	
	
	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		Initialize(void)
	{
		m_Image->Initialize();
	}


	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		FillBuffer (const TPixel& value)
	{
		m_Image->FillBuffer(value);
	}


	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		SetPixel(const IndexType &index, const TPixel& value)
	{
		m_Image->SetPixel(
			ConvertToStaticArray<IndexType, IndexTypeD>::DO(index),
			value );
	}


	template < class TPixel, unsigned int VDimension >
		const TPixel& NDImageTemplate<TPixel, VDimension>::
		GetPixel(const IndexType &index) const
	{
		return m_Image->GetPixel(
			ConvertToStaticArray<IndexType, IndexTypeD>::DO(index)  );
	}


	template < class TPixel, unsigned int VDimension >
		TPixel& NDImageTemplate<TPixel, VDimension>::
		GetPixel(const IndexType &index)
	{
		return m_Image->GetPixel(
			ConvertToStaticArray<IndexType, IndexTypeD>::DO(index)	);
	}


	template < class TPixel, unsigned int VDimension >
		TPixel * NDImageTemplate<TPixel, VDimension>::
		GetBufferPointer()
	{
		return m_Image->GetBufferPointer();
	}


	template < class TPixel, unsigned int VDimension >
		const TPixel * NDImageTemplate<TPixel, VDimension>::
		GetBufferPointer() const
	{
		return m_Image->GetBufferPointer();
	}


	template < class TPixel, unsigned int VDimension >
		typename NDImageTemplate<TPixel, VDimension>::PixelContainer *
		NDImageTemplate<TPixel, VDimension>::
		GetPixelContainer()
	{
		return m_Image->GetPixelContainer();
	}


	template < class TPixel, unsigned int VDimension >
		const typename NDImageTemplate<TPixel, VDimension>::PixelContainer *
		NDImageTemplate<TPixel, VDimension>::
		GetPixelContainer() const
	{
		return m_Image->GetPixelContainer();
	}


	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		SetPixelContainer( PixelContainer *container )
	{
		m_Image->SetPixelContainer(container);
	}


	template < class TPixel, unsigned int VDimension >
		typename NDImageTemplate<TPixel, VDimension>::AccessorType
		NDImageTemplate<TPixel, VDimension>::
		GetPixelAccessor( void )
	{
		return m_Image->GetPixelAccessor();
	}


	template < class TPixel, unsigned int VDimension >
		const typename NDImageTemplate<TPixel, VDimension>::AccessorType
		NDImageTemplate<TPixel, VDimension>::
		GetPixelAccessor( void ) const
	{
		return m_Image->GetPixelAccessor();
	}


	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		SetSpacing( const SpacingType & spacing )
	{
		m_Image->SetSpacing( 
			ConvertToStaticArray<SpacingType, SpacingTypeD>::DO(spacing)	);			
	}


	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		SetOrigin( const PointType & origin )
	{
		m_Image->SetOrigin( 
			ConvertToStaticArray<PointType, PointTypeD>::DO(origin)	 );			
	}


	template < class TPixel, unsigned int VDimension >
		typename NDImageTemplate<TPixel, VDimension>::SpacingType
		NDImageTemplate<TPixel, VDimension>::
		GetSpacing(void)
	{
		return ConvertToDynamicArray<SpacingTypeD, SpacingType>::DO(
			m_Image->GetSpacing()   );
	}


	template < class TPixel, unsigned int VDimension >
		typename NDImageTemplate<TPixel, VDimension>::PointType
		NDImageTemplate<TPixel, VDimension>::
		GetOrigin(void)
	{
		return ConvertToDynamicArray<PointTypeD, PointType>::DO(
			m_Image->GetOrigin()   );
	}


	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		CopyInformation(const DataObject *data)
	{
		m_Image->CopyInformation(data);
	}


	template < class TPixel, unsigned int VDimension >
		const typename NDImageTemplate<TPixel, VDimension>::OffsetValueType *
		NDImageTemplate<TPixel, VDimension>::
		GetOffsetTable() const
	{
		return m_Image->GetOffsetTable();
	}


	template < class TPixel, unsigned int VDimension >
		typename NDImageTemplate<TPixel, VDimension>::OffsetValueType
		NDImageTemplate<TPixel, VDimension>::
		ComputeOffset(const IndexType &ind) const
	{
		return m_Image->ComputeOffset(
			ConvertToStaticArray<IndexType, IndexTypeD>::DO(ind)   );
	}

	
	template < class TPixel, unsigned int VDimension >
		typename NDImageTemplate<TPixel, VDimension>::IndexType
		NDImageTemplate<TPixel, VDimension>::
		ComputeIndex(OffsetValueType offset) const
	{
		return ConvertToDynamicArray<IndexTypeD, IndexType>::DO(
			m_Image->ComputeIndex(offset )   );
	}
	
	
	template < class TPixel, unsigned int VDimension >
		unsigned int  NDImageTemplate<TPixel, VDimension>::
		ImageDimension(void)
	{
		return m_Image->GetImageDimension();
	}
			
		
	template < class TPixel, unsigned int VDimension >
		unsigned int NDImageTemplate<TPixel, VDimension>::
		GetImageDimension(void)
	{
		return m_Image->GetImageDimension();
	}
	 
		 
	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		Write(void)
	{
		if (m_Writer)
		{
			m_Writer->SetInput(m_Image);
			m_Writer->Write(); 
		}
	}	 
		

	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		Read(void)
	{
		if (m_Reader)
		{
			m_Reader->Update();
			m_Image = m_Reader->GetOutput(); 
		}
	}	 

	
	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		CreateNewImage(void)
	{
		m_Image = ImageType::New();
	}	 


	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		SetImageIOWriter (ImageIOBase *_arg)
	{
		if (!m_Writer)
		{
			m_Writer=WriterType::New();
		}
		m_Writer->SetImageIO(_arg);
	}	 


	template < class TPixel, unsigned int VDimension >
		ImageIOBase *	NDImageTemplate<TPixel, VDimension>::
		GetImageIOWriter(void)
	{
		if (m_Writer)
		{
			return m_Writer->GetImageIO();
		}
		else
		{
			return 0;
		}
	}	 


	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		SetImageIOReader (ImageIOBase *_arg)
	{
		if (!m_Reader)
		{
			m_Reader=ReaderType::New();
		}
		m_Reader->SetImageIO(_arg);
	}	 


	template < class TPixel, unsigned int VDimension >
		ImageIOBase * NDImageTemplate<TPixel, VDimension>::
		GetImageIOReader(void)
	{
		if (m_Reader)
		{
			return m_Reader->GetImageIO();
		}
		else
		{
			return 0;
		}
	}	 

	
	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		SetOutputFileName(const char * name)
	{
		if (!m_Writer)
		{
			m_Writer=WriterType::New();
		}
		m_Writer->SetFileName(name);
	}	 

		
	template < class TPixel, unsigned int VDimension >
		void NDImageTemplate<TPixel, VDimension>::
		SetInputFileName(const char * name )
	{
		if (!m_Reader)
		{
			m_Reader=ReaderType::New();
		}
		m_Reader->SetFileName(name);
	}	 

		
	template < class TPixel, unsigned int VDimension >
		const char * NDImageTemplate<TPixel, VDimension>::
		GetOutputFileName(void)
	{
		if (m_Writer) 
		{
			return m_Writer->GetFileName();
		}
		else
		{
			return "";
		}
	}	 

		
	template < class TPixel, unsigned int VDimension >
		const char * NDImageTemplate<TPixel, VDimension>::
		GetInputFileName(void)
	{
		if (m_Reader) 
		{
			return m_Reader->GetFileName();
		}
		else
		{
			return "";
		}
	}	 

				  	
} // end namespace itk


#endif // end #ifndef __itkNDImageTemplate_hxx

